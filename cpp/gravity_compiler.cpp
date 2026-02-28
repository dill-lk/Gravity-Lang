#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cstdlib>
#include <filesystem>

struct Body {
    std::string name;
    std::array<double, 3> pos{0, 0, 0};
    std::array<double, 3> vel{0, 0, 0};
    double mass = 0.0;
    bool fixed = false;
};

struct Program {
    std::vector<Body> bodies;
    std::vector<std::pair<std::string, std::string>> pulls;
    std::vector<std::string> print_position_names;
    std::vector<std::string> print_velocity_names;
    struct ObserveSpec {
        std::string name;
        std::string output_file;
        int frequency = 1;
    };
    struct ThrustSpec {
        std::string name;
        std::array<double, 3> delta_v{0, 0, 0};
    };
    std::vector<ObserveSpec> observe_specs;
    std::vector<ThrustSpec> step_thrusts;
    int dump_all_frequency = 0;
    std::string dump_all_file;
    bool auto_plot = false;
    std::string auto_plot_body = "Rocket";
    std::string integrator = "verlet";
    double friction = 0.0;
    int steps = 0;
    double dt = 1.0;
};

static inline std::string trim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
    return s;
}

static double unit_scale(const std::string& unit) {
    if (unit == "m" || unit == "m/s" || unit == "kg" || unit == "s") return 1.0;
    if (unit == "km" || unit == "km/s") return 1000.0;
    if (unit == "min") return 60.0;
    if (unit == "hour") return 3600.0;
    if (unit == "day" || unit == "days") return 86400.0;
    return 1.0;
}

static std::array<double, 3> parse_vec(const std::string& value, const std::string& unit) {
    std::regex v(R"(\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\])");
    std::smatch m;
    if (!std::regex_search(value, m, v)) throw std::runtime_error("invalid vector: " + value);
    const double s = unit_scale(unit);
    return {std::stod(m[1]) * s, std::stod(m[2]) * s, std::stod(m[3]) * s};
}

static Program parse_gravity(const std::string& script_path, bool strict_mode = false) {
    Program p;
    std::ifstream in(script_path);
    if (!in) throw std::runtime_error("cannot open script: " + script_path);

    std::regex sphere_head_re(R"(^(sphere|probe|rocket)\s+(\w+)\s+at\s+(\[[^\]]+\])(?:\[(\w+)\])?.*$)");
    std::regex mass_re(R"(mass\s+([-+0-9.eE]+)\[(\w+)\])");
    std::regex velocity_re(R"(velocity\s+(\[[^\]]+\])\[(\w+\/s)\])");
    std::regex velocity_assign_re(R"(^([A-Za-z_]\w*)\.velocity\s*=\s*(\[[^\]]+\])\[(\w+\/s)\]\s*$)");
    std::regex orbit_re(R"((orbit|simulate)\s+\w+\s+in\s+([-+0-9.eE]+)\.\.([-+0-9.eE]+)\s+(dt|step)\s+([-+0-9.eE]+)\[(\w+)\].*\{)");
    std::regex print_re(R"(^print\s+(\w+)\.position\s*$)");
    std::regex print_vel_re(R"(^print\s+(\w+)\.velocity\s*$)");
    std::regex grav_all_re(R"(^grav\s+all\s*$)");
    std::regex friction_re(R"(^friction\s+([-+0-9.eE]+)\s*$)");
    std::regex thrust_re(R"(^thrust\s+([A-Za-z_]\w*)\s+by\s+(\[[^\]]+\])\[(\w+\/s)\]\s*$)");
    std::regex observe_re(R"obs(^observe\s+([A-Za-z_]\w*)\.position\s+to\s+"([^"]+)"\s+frequency\s+(\d+)\s*$)obs");
    std::regex dump_all_re(R"obs(^dump_all\s+to\s+"([^"]+)"\s+frequency\s+(\d+)\s*$)obs");
    std::regex plot_re(R"(^plot\s+(on|off)(?:\s+body\s+([A-Za-z_]\w*))?\s*$)");
    std::regex verbose_re(R"(^verbose\s+(on|off)\s*$)");
    std::regex merge_heat_re(R"(^merge_heat\s+([-+0-9.eE]+)\s*$)");
    std::regex throttle_target_re(R"(^throttle\s+([A-Za-z_]\w*)\s+to\s+maintain\s+velocity\s+([-+0-9.eE]+)\[(m\/s)\]\s*$)");
    std::regex gravity_turn_re(R"(^gravity_turn\s+([A-Za-z_]\w*)\s+start\s+([-+0-9.eE]+)\[(m|km)\]\s+end\s+([-+0-9.eE]+)\[(m|km)\]\s+final_pitch\s+([-+0-9.eE]+)\s*$)");
    std::regex detach_event_re(R"(^event\s+step\s+(\d+)\s+detach\s+([A-Za-z_]\w*)\s+from\s+([A-Za-z_]\w*)\s*$)");
    std::regex rocket_assign_re(R"(^([A-Za-z_]\w*)\.(dry_mass|fuel_mass|burn_rate|max_thrust|isp_sea_level|isp_vacuum|drag_coefficient|cross_section|throttle)\s*=.*$)");

    bool in_block = false;
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (const auto hash = line.find('#'); hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        std::smatch m;
        if (std::regex_match(line, m, sphere_head_re)) {
            Body b;
            b.name = m[2];
            b.pos = parse_vec(m[3], m[4].matched ? m[4].str() : "m");

            std::smatch mass_match;
            if (!std::regex_search(line, mass_match, mass_re)) {
                throw std::runtime_error("sphere missing mass: " + line);
            }
            b.mass = std::stod(mass_match[1]) * unit_scale(mass_match[2]);

            std::smatch vel_match;
            if (std::regex_search(line, vel_match, velocity_re)) {
                b.vel = parse_vec(vel_match[1], vel_match[2]);
            }
            if (line.find(" fixed") != std::string::npos) b.fixed = true;
            p.bodies.push_back(b);
            continue;
        }

        if (std::regex_match(line, m, velocity_assign_re)) {
            const std::string target = m[1];
            for (auto& b : p.bodies) {
                if (b.name == target) {
                    b.vel = parse_vec(m[2], m[3]);
                    break;
                }
            }
            continue;
        }

        if (std::regex_match(line, m, orbit_re)) {
            const double start = std::stod(m[2]);
            const double stop = std::stod(m[3]);
            p.steps = std::max(1, static_cast<int>(std::round(stop - start)));
            p.dt = std::stod(m[5]) * unit_scale(m[6]);
            std::regex integrator_re(R"(integrator\s+([A-Za-z_][A-Za-z0-9_]*))");
            std::smatch integrator_match;
            if (std::regex_search(line, integrator_match, integrator_re)) {
                p.integrator = integrator_match[1];
            }
            in_block = true;
            continue;
        }

        if (std::regex_match(line, m, friction_re)) {
            p.friction = std::stod(m[1]);
            continue;
        }

        if (std::regex_match(line, grav_all_re)) {
            for (const auto& s : p.bodies) {
                for (const auto& t : p.bodies) {
                    if (s.name != t.name) p.pulls.push_back({s.name, t.name});
                }
            }
            continue;
        }

        if (in_block && line == "}") {
            in_block = false;
            continue;
        }

        if (line.find(" pull ") != std::string::npos) {
            auto pos = line.find(" pull ");
            std::string src = trim(line.substr(0, pos));
            std::string targets = trim(line.substr(pos + 6));
            std::stringstream ss(targets);
            std::string t;
            while (std::getline(ss, t, ',')) {
                t = trim(t);
                if (!t.empty()) p.pulls.push_back({src, t});
            }
            continue;
        }

        if (in_block && std::regex_match(line, m, print_re)) {
            p.print_position_names.push_back(m[1]);
            continue;
        }

        if (in_block && std::regex_match(line, m, print_vel_re)) {
            p.print_velocity_names.push_back(m[1]);
            continue;
        }

        if (std::regex_match(line, m, thrust_re)) {
            Program::ThrustSpec thrust;
            thrust.name = m[1];
            thrust.delta_v = parse_vec(m[2], m[3]);
            p.step_thrusts.push_back(thrust);
            continue;
        }

        if (std::regex_match(line, m, dump_all_re)) {
            p.dump_all_file = m[1];
            p.dump_all_frequency = std::max(1, std::stoi(m[2]));
            continue;
        }

        if (std::regex_match(line, m, plot_re)) {
            p.auto_plot = (m[1] == "on");
            if (m[2].matched) p.auto_plot_body = m[2];
            continue;
        }

        if (in_block && std::regex_match(line, m, observe_re)) {
            Program::ObserveSpec obs;
            obs.name = m[1];
            obs.output_file = m[2];
            obs.frequency = std::max(1, std::stoi(m[3]));
            p.observe_specs.push_back(obs);
            continue;
        }

        if (std::regex_match(line, m, verbose_re) ||
            std::regex_match(line, m, merge_heat_re) ||
            std::regex_match(line, m, throttle_target_re) ||
            std::regex_match(line, m, gravity_turn_re) ||
            std::regex_match(line, m, detach_event_re) ||
            std::regex_match(line, m, rocket_assign_re)) {
            continue;
        }

        if (strict_mode) throw std::runtime_error("unsupported line " + std::to_string(line_no) + ": " + line);
    }

    if (p.steps <= 0) throw std::runtime_error("no orbit/simulate loop found");
    if (p.bodies.empty()) throw std::runtime_error("no sphere objects found");
    if (p.pulls.empty() && p.bodies.size() > 1) {
        for (const auto& s : p.bodies) {
            for (const auto& t : p.bodies) {
                if (s.name != t.name) p.pulls.push_back({s.name, t.name});
            }
        }
    }
    return p;
}

static std::string emit_cpp(const Program& p) {
    std::ostringstream out;
    out << "#include <cmath>\n#include <fstream>\n#include <iostream>\n#include <string>\n#include <unordered_map>\n#include <vector>\n#include <array>\n#include <filesystem>\n"
        << "struct Body{std::string n;double px,py,pz,vx,vy,vz,m;bool f;};\n"
        << "int main(){const double G=6.67430e-11;std::vector<Body>b={\n";
    for (const auto& body : p.bodies) {
        out << "{\"" << body.name << "\"," << body.pos[0] << "," << body.pos[1] << "," << body.pos[2]
            << "," << body.vel[0] << "," << body.vel[1] << "," << body.vel[2] << "," << body.mass
            << "," << (body.fixed ? "true" : "false") << "},\n";
    }
    out << "};std::unordered_map<std::string,int>ix;for(int i=0;i<(int)b.size();++i)ix[b[i].n]=i;\n";
    out << "auto ensure_parent_dir=[&](const std::string& p){auto parent=std::filesystem::path(p).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);};\n";
    out << "std::vector<std::pair<int,int>>pulls={";
    for (const auto& pr : p.pulls) out << "{" << "ix[\"" << pr.first << "\"],ix[\"" << pr.second << "\"]},";
    out << "};\n";

    for (size_t i = 0; i < p.observe_specs.size(); ++i) {
        const auto& obs = p.observe_specs[i];
        out << "ensure_parent_dir(\"" << obs.output_file << "\");std::ofstream obs" << i << "(\"" << obs.output_file << "\");obs" << i << "<<\"step,x,y,z\\n\";\n";
    }

    if (p.dump_all_frequency > 0 && !p.dump_all_file.empty()) {
        out << "ensure_parent_dir(\"" << p.dump_all_file << "\");std::ofstream dump_all(\"" << p.dump_all_file << "\");dump_all<<\"step,body,x,y,z,vx,vy,vz\\n\";\n";
    }

    out << "auto compute_acc=[&](const std::vector<Body>&state){std::vector<std::array<double,3>>a(state.size(),{0,0,0});"
        << "for(auto &pr:pulls){auto&s=state[pr.first];auto&t=state[pr.second];double dx=s.px-t.px,dy=s.py-t.py,dz=s.pz-t.pz;double r2=dx*dx+dy*dy+dz*dz+1e-18;double r=sqrt(r2);double am=G*s.m/r2;a[pr.second][0]+=am*dx/r;a[pr.second][1]+=am*dy/r;a[pr.second][2]+=am*dz/r;}return a;};\n";

    out << "for(int step=0;step<" << p.steps << ";++step){\n";

    for (const auto& thrust : p.step_thrusts) {
        out << "{auto &o=b[ix[\"" << thrust.name << "\"]];if(!o.f){o.vx+=" << thrust.delta_v[0]
            << ";o.vy+=" << thrust.delta_v[1] << ";o.vz+=" << thrust.delta_v[2] << ";}}\n";
    }

    if (p.integrator == "euler") {
        out << "auto a=compute_acc(b);for(size_t i=0;i<b.size();++i){if(b[i].f)continue;"
            << "b[i].vx+=a[i][0]*" << p.dt << ";b[i].vy+=a[i][1]*" << p.dt << ";b[i].vz+=a[i][2]*" << p.dt << ";"
            << "b[i].vx*=(1.0-" << p.friction << ");b[i].vy*=(1.0-" << p.friction << ");b[i].vz*=(1.0-" << p.friction << ");"
            << "b[i].px+=b[i].vx*" << p.dt << ";b[i].py+=b[i].vy*" << p.dt << ";b[i].pz+=b[i].vz*" << p.dt << ";}\n";
    } else if (p.integrator == "leapfrog") {
        out << "auto a1=compute_acc(b);std::vector<std::array<double,3>>half(b.size(),{0,0,0});"
            << "for(size_t i=0;i<b.size();++i){if(b[i].f)continue;half[i][0]=b[i].vx+a1[i][0]*" << p.dt * 0.5
            << ";half[i][1]=b[i].vy+a1[i][1]*" << p.dt * 0.5 << ";half[i][2]=b[i].vz+a1[i][2]*" << p.dt * 0.5
            << ";b[i].px+=half[i][0]*" << p.dt << ";b[i].py+=half[i][1]*" << p.dt << ";b[i].pz+=half[i][2]*" << p.dt << ";}"
            << "auto a2=compute_acc(b);for(size_t i=0;i<b.size();++i){if(b[i].f)continue;b[i].vx=half[i][0]+a2[i][0]*" << p.dt * 0.5
            << ";b[i].vy=half[i][1]+a2[i][1]*" << p.dt * 0.5 << ";b[i].vz=half[i][2]+a2[i][2]*" << p.dt * 0.5
            << ";b[i].vx*=(1.0-" << p.friction << ");b[i].vy*=(1.0-" << p.friction << ");b[i].vz*=(1.0-" << p.friction << ");}\n";
    } else {
        out << "auto a1=compute_acc(b);std::vector<Body>tmp=b;for(size_t i=0;i<b.size();++i){if(b[i].f)continue;"
            << "tmp[i].px=b[i].px+b[i].vx*" << p.dt << "+0.5*a1[i][0]*" << p.dt * p.dt << ";"
            << "tmp[i].py=b[i].py+b[i].vy*" << p.dt << "+0.5*a1[i][1]*" << p.dt * p.dt << ";"
            << "tmp[i].pz=b[i].pz+b[i].vz*" << p.dt << "+0.5*a1[i][2]*" << p.dt * p.dt << ";}"
            << "auto a2=compute_acc(tmp);for(size_t i=0;i<b.size();++i){if(b[i].f)continue;"
            << "b[i].px=tmp[i].px;b[i].py=tmp[i].py;b[i].pz=tmp[i].pz;"
            << "b[i].vx+=(a1[i][0]+a2[i][0])*" << p.dt * 0.5 << ";b[i].vy+=(a1[i][1]+a2[i][1])*" << p.dt * 0.5
            << ";b[i].vz+=(a1[i][2]+a2[i][2])*" << p.dt * 0.5 << ";"
            << "b[i].vx*=(1.0-" << p.friction << ");b[i].vy*=(1.0-" << p.friction << ");b[i].vz*=(1.0-" << p.friction << ");}\n";
    }

    for (const auto& name : p.print_position_names) {
        out << "{auto&o=b[ix[\"" << name << "\"]];std::cout<<\"" << name
            << ".position=(\"<<o.px<<\", \"<<o.py<<\", \"<<o.pz<<\")\\n\";}\n";
    }

    for (const auto& name : p.print_velocity_names) {
        out << "{auto&o=b[ix[\"" << name << "\"]];std::cout<<\"" << name
            << ".velocity=(\"<<o.vx<<\", \"<<o.vy<<\", \"<<o.vz<<\")\\n\";}\n";
    }

    for (size_t i = 0; i < p.observe_specs.size(); ++i) {
        const auto& obs = p.observe_specs[i];
        out << "if((step+1)%" << obs.frequency << "==0){auto&o=b[ix[\"" << obs.name << "\"]];"
            << "obs" << i << "<<(step+1)<<\",\"<<o.px<<\",\"<<o.py<<\",\"<<o.pz<<\"\\n\";}\n";
    }

    if (p.dump_all_frequency > 0 && !p.dump_all_file.empty()) {
        out << "if((step+1)%" << p.dump_all_frequency << "==0){for(const auto&o:b){dump_all<<(step+1)<<\",\"<<o.n<<\",\"<<o.px<<\",\"<<o.py<<\",\"<<o.pz<<\",\"<<o.vx<<\",\"<<o.vy<<\",\"<<o.vz<<\"\\n\";}}\n";
    }

    out << "}\n";
    if (p.auto_plot && p.dump_all_frequency > 0 && !p.dump_all_file.empty()) {
        out << "std::cout<<\"\\n[System] Simulation complete. Launching telemetry dashboard...\\n\";"
            << "std::system(\"python3 tools/telemetry_dashboard.py \\\"" << p.dump_all_file << "\\\" --body \\\"" << p.auto_plot_body
            << "\\\" || python tools/telemetry_dashboard.py \\\"" << p.dump_all_file << "\\\" --body \\\"" << p.auto_plot_body << "\\\"\");\n";
    }
    out << "return 0;}\n";
    return out.str();
}

int main(int argc, char** argv) {
    auto print_help = []() {
        std::cout
            << "==============================\n"
            << "    GRAVITY-LANG C++ EMITTER  \n"
            << "==============================\n"
            << "usage:\n"
            << "  gravityc <script.gravity> --emit <out.cpp> [--build <exe>] [--run] [--cxx <compiler>] [--strict]\n"
            << "  gravityc --help\n"
            << "  gravityc --version\n";
    };

    if (argc < 2) {
        print_help();
        std::cout << "\nTip: use `gravityc <script.gravity> --emit out.cpp` to generate C++.\n";
        return 2;
    }

    const std::string first = argv[1];
    if (first == "--help" || first == "-h" || first == "help") {
        print_help();
        return 0;
    }
    if (first == "--version" || first == "version") {
        std::cout << "gravityc ENGINE v3.0 emitter build " << __DATE__ << " " << __TIME__ << "\n";
        return 0;
    }

    std::string script = argv[1];
    std::string emit;
    std::string build;
    std::string cxx = "g++";
    bool run_output = false;
    bool strict_mode = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--emit" && i + 1 < argc) emit = argv[++i];
        else if (arg == "--build" && i + 1 < argc) build = argv[++i];
        else if (arg == "--run") run_output = true;
        else if (arg == "--cxx" && i + 1 < argc) cxx = argv[++i];
        else if (arg == "--strict") strict_mode = true;
        else {
            std::cerr << "error: unknown option: " << arg << "\n";
            print_help();
            return 2;
        }
    }

    if (emit.empty()) {
        std::cerr << "error: --emit is required\n";
        print_help();
        return 2;
    }

    try {
        Program p = parse_gravity(script, strict_mode);
        std::ofstream out(emit);
        if (!out) throw std::runtime_error("cannot open output file: " + emit);
        out << emit_cpp(p);
        out.close();

        if (!build.empty()) {
            std::string cmd = cxx + " -O3 -std=c++17 ";
#ifdef _WIN32
            cmd += "-static -static-libstdc++ -static-libgcc ";
#endif
            cmd += "\"" + emit + "\" -o \"" + build + "\"";
            int rc = std::system(cmd.c_str());
            if (rc != 0) throw std::runtime_error("compiler failed: " + cmd);
            std::cout << "built executable: " << build << "\n";
            if (run_output) {
                const std::string run_cmd = "\"" + build + "\"";
                const int run_rc = std::system(run_cmd.c_str());
                if (run_rc != 0) throw std::runtime_error("built executable failed: " + run_cmd);
            }
        } else if (run_output) {
            throw std::runtime_error("--run requires --build <exe>");
        }
        std::cout << "emitted C++: " << emit << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
