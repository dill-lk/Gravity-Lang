#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Vec3 = std::array<double, 3>;

enum class ObserveKind { Position, Velocity };
enum class GravityModel { Newtonian, Mond, GRCorrection };

struct Body {
    std::string name;
    Vec3 pos{0.0, 0.0, 0.0};
    Vec3 vel{0.0, 0.0, 0.0};
    double mass = 0.0;
    double radius = 0.0;
    bool fixed = false;
};

struct ObserveSpec {
    std::string name;
    std::string output_file;
    int frequency = 1;
    ObserveKind kind = ObserveKind::Position;
};

struct ThrustSpec {
    std::string name;
    Vec3 delta_v{0.0, 0.0, 0.0};
};

struct OrbitalRequest {
    std::string body;
    std::string center;
    bool before_sim = true;
};

struct Program {
    std::vector<Body> bodies;
    std::vector<std::pair<std::string, std::string>> pulls;
    std::vector<std::string> print_position_names;
    std::vector<std::string> print_velocity_names;
    std::vector<ObserveSpec> observe_specs;
    std::vector<ThrustSpec> step_thrusts;
    std::vector<OrbitalRequest> orbital_requests;
    std::string integrator = "verlet";
    double friction = 0.0;
    int steps = 0;
    double dt = 1.0;
    bool collisions_on = false;
    bool monitor_energy = false;
    bool monitor_momentum = false;
    bool monitor_angular_momentum = false;
    bool adaptive_on = false;
    double adaptive_tol = 1e-3;
    double adaptive_dt_min = 1.0;
    double adaptive_dt_max = 3600.0;
    double softening = 1e-9;
    GravityModel gravity_model = GravityModel::Newtonian;
    double mond_a0 = 1.2e-10;
    double gr_beta = 1e16;
    double gravity_constant = 6.67430e-11;
    unsigned int worker_threads = 0;
    size_t threading_min_interactions = 2048;
    bool profile_on = false;
};

std::string trim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
    return s;
}

double unit_scale(const std::string& unit) {
    if (unit == "m" || unit == "m/s" || unit == "kg" || unit == "s") return 1.0;
    if (unit == "km" || unit == "km/s") return 1000.0;
    if (unit == "min") return 60.0;
    if (unit == "hour") return 3600.0;
    if (unit == "day" || unit == "days") return 86400.0;
    throw std::runtime_error("unsupported unit: " + unit);
}

Vec3 parse_vec(const std::string& value, const std::string& unit) {
    const auto lb = value.find('[');
    const auto rb = value.rfind(']');
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb + 1) {
        throw std::runtime_error("invalid vector: " + value);
    }
    std::string inside = value.substr(lb + 1, rb - lb - 1);
    std::array<std::string, 3> comps{};
    size_t start = 0;
    for (int i = 0; i < 3; ++i) {
        const size_t comma = (i < 2) ? inside.find(',', start) : std::string::npos;
        if (i < 2 && comma == std::string::npos) throw std::runtime_error("invalid vector: " + value);
        comps[static_cast<size_t>(i)] = trim(inside.substr(start, (comma == std::string::npos ? inside.size() : comma) - start));
        start = (comma == std::string::npos) ? inside.size() : comma + 1;
    }
    if (trim(inside.substr(start)).size() > 0) throw std::runtime_error("invalid vector: " + value);

    const double s = unit_scale(unit);
    return {static_cast<double>(std::stold(comps[0]) * s),
            static_cast<double>(std::stold(comps[1]) * s),
            static_cast<double>(std::stold(comps[2]) * s)};
}

double mag(const Vec3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

double total_energy(const std::vector<Body>& bodies) {
    constexpr long double G = 6.67430e-11L;
    long double kinetic = 0.0L;
    for (const auto& b : bodies) {
        const long double v2 = static_cast<long double>(b.vel[0]) * b.vel[0] + static_cast<long double>(b.vel[1]) * b.vel[1] + static_cast<long double>(b.vel[2]) * b.vel[2];
        kinetic += 0.5L * static_cast<long double>(b.mass) * v2;
    }

    long double potential = 0.0L;
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            const auto d = sub(bodies[i].pos, bodies[j].pos);
            const double r = std::max(1e-9, mag(d));
            potential += -G * static_cast<long double>(bodies[i].mass) * static_cast<long double>(bodies[j].mass) / r;
        }
    }
    return static_cast<double>(kinetic + potential);
}

Vec3 total_momentum(const std::vector<Body>& bodies) {
    long double px = 0.0L;
    long double py = 0.0L;
    long double pz = 0.0L;
    for (const auto& b : bodies) {
        px += static_cast<long double>(b.mass) * static_cast<long double>(b.vel[0]);
        py += static_cast<long double>(b.mass) * static_cast<long double>(b.vel[1]);
        pz += static_cast<long double>(b.mass) * static_cast<long double>(b.vel[2]);
    }
    return {static_cast<double>(px), static_cast<double>(py), static_cast<double>(pz)};
}

Vec3 center_of_mass(const std::vector<Body>& bodies) {
    long double m = 0.0L;
    long double x = 0.0L;
    long double y = 0.0L;
    long double z = 0.0L;
    for (const auto& b : bodies) {
        m += static_cast<long double>(b.mass);
        x += static_cast<long double>(b.mass) * static_cast<long double>(b.pos[0]);
        y += static_cast<long double>(b.mass) * static_cast<long double>(b.pos[1]);
        z += static_cast<long double>(b.mass) * static_cast<long double>(b.pos[2]);
    }
    const long double inv = (m > 0.0L) ? 1.0L / m : 0.0L;
    return {static_cast<double>(x * inv), static_cast<double>(y * inv), static_cast<double>(z * inv)};
}

Vec3 total_angular_momentum(const std::vector<Body>& bodies) {
    long double lx = 0.0L;
    long double ly = 0.0L;
    long double lz = 0.0L;
    for (const auto& b : bodies) {
        const long double px = static_cast<long double>(b.mass) * static_cast<long double>(b.vel[0]);
        const long double py = static_cast<long double>(b.mass) * static_cast<long double>(b.vel[1]);
        const long double pz = static_cast<long double>(b.mass) * static_cast<long double>(b.vel[2]);
        const long double rx = static_cast<long double>(b.pos[0]);
        const long double ry = static_cast<long double>(b.pos[1]);
        const long double rz = static_cast<long double>(b.pos[2]);
        lx += ry * pz - rz * py;
        ly += rz * px - rx * pz;
        lz += rx * py - ry * px;
    }
    return {static_cast<double>(lx), static_cast<double>(ly), static_cast<double>(lz)};
}

void apply_collisions(std::vector<Body>& bodies) {
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            const double collision_radius = bodies[i].radius + bodies[j].radius;
            if (collision_radius <= 0.0) continue;
            const auto d = sub(bodies[i].pos, bodies[j].pos);
            if (mag(d) <= collision_radius) {
                if (!bodies[i].fixed) bodies[i].vel = {0.0, 0.0, 0.0};
                if (!bodies[j].fixed) bodies[j].vel = {0.0, 0.0, 0.0};
            }
        }
    }
}

void print_orbital_elements(const std::vector<Body>& b, int object_idx, int center_idx) {
    constexpr long double G = 6.67430e-11L;
    const Body& obj = b[static_cast<size_t>(object_idx)];
    const Body& center = b[static_cast<size_t>(center_idx)];

    Vec3 r = sub(obj.pos, center.pos);
    Vec3 v = sub(obj.vel, center.vel);

    const double rmag = mag(r);
    const double vmag = mag(v);
    const double mu = G * center.mass;
    const double energy = 0.5 * vmag * vmag - mu / std::max(1e-12, rmag);
    const double a = std::abs(energy) > 1e-12 ? -mu / (2.0 * energy) : std::numeric_limits<double>::infinity();

    const Vec3 h = {
        r[1] * v[2] - r[2] * v[1],
        r[2] * v[0] - r[0] * v[2],
        r[0] * v[1] - r[1] * v[0],
    };
    const double h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
    const double e = std::sqrt(std::max(0.0, 1.0 + (2.0 * energy * h2) / (mu * mu)));

    std::cout << "orbital_elements " << obj.name << " around " << center.name
              << ": semi_major_axis=" << a
              << " m, eccentricity=" << e << "\n";
}

Program parse_gravity(const std::string& script_path, bool strict_mode = false) {
    Program p;
    std::ifstream in(script_path);
    if (!in) throw std::runtime_error("cannot open script: " + script_path);

    std::regex sphere_head_re(R"(^(sphere|probe)\s+(\w+)\s+at\s+(\[[^\]]+\])(?:\[(\w+)\])?.*$)");
    std::regex mass_re(R"(mass\s+([-+0-9.eE]+)\[(\w+)\])");
    std::regex radius_re(R"(radius\s+([-+0-9.eE]+)\[(\w+)\])");
    std::regex velocity_re(R"(velocity\s+(\[[^\]]+\])\[(\w+\/s)\])");
    std::regex velocity_assign_re(R"(^([A-Za-z_]\w*)\.velocity\s*=\s*(\[[^\]]+\])\[(\w+\/s)\]\s*$)");
    std::regex orbit_re(R"((orbit|simulate)\s+\w+\s+in\s+([-+0-9.eE]+)\.\.([-+0-9.eE]+)\s+(dt|step)\s+([-+0-9.eE]+)\[(\w+)\].*\{)");
    std::regex print_re(R"(^print\s+(\w+)\.(position|velocity)\s*$)");
    std::regex grav_all_re(R"(^grav\s+all\s*$)");
    std::regex friction_re(R"(^friction\s+([-+0-9.eE]+)\s*$)");
    std::regex thrust_re(R"(^thrust\s+([A-Za-z_]\w*)\s+by\s+(\[[^\]]+\])\[(\w+\/s)\]\s*$)");
    std::regex observe_re(R"obs(^observe\s+([A-Za-z_]\w*)\.(position|velocity)\s+to\s+"([^"]+)"\s+frequency\s+(\d+)\s*$)obs");
    std::regex orbital_re(R"(^orbital_elements\s+([A-Za-z_]\w*)\s+around\s+([A-Za-z_]\w*)\s*$)");
    std::regex step_physics_re(R"(^step_physics\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)\s*$)");
    std::regex adaptive_re(R"(^adaptive\s+tol\s+([-+0-9.eE]+)\s+min\s+([-+0-9.eE]+)\[(\w+)\]\s+max\s+([-+0-9.eE]+)\[(\w+)\]\s*$)");
    std::regex softening_re(R"(^softening\s+([-+0-9.eE]+)\[(\w+)\]\s*$)");
    std::regex gravity_model_re(R"(^gravity_model\s+(newtonian|mond|gr_correction)(?:\s+a0\s+([-+0-9.eE]+)\[(m\/s2)\])?(?:\s+beta\s+([-+0-9.eE]+))?\s*$)");
    std::regex gravity_constant_re(R"(^gravity_constant\s+([-+0-9.eE]+)\s*$)");
    std::regex threads_re(R"(^threads\s+(auto|\d+)\s*$)");
    std::regex threading_min_re(R"(^threading\s+min_interactions\s+(\d+)\s*$)");
    std::regex profile_re(R"(^profile\s+(on|off)\s*$)");

    bool in_block = false;
    bool has_sim = false;
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
            if (!std::regex_search(line, mass_match, mass_re)) throw std::runtime_error("sphere missing mass: " + line);
            b.mass = std::stod(mass_match[1]) * unit_scale(mass_match[2]);

            std::smatch radius_match;
            if (std::regex_search(line, radius_match, radius_re)) {
                b.radius = std::stod(radius_match[1]) * unit_scale(radius_match[2]);
            }

            std::smatch vel_match;
            if (std::regex_search(line, vel_match, velocity_re)) b.vel = parse_vec(vel_match[1], vel_match[2]);
            if (line.find(" fixed") != std::string::npos) b.fixed = true;
            p.bodies.push_back(b);
            continue;
        }

        if (std::regex_match(line, m, velocity_assign_re)) {
            bool found = false;
            for (auto& b : p.bodies) {
                if (b.name == m[1]) {
                    b.vel = parse_vec(m[2], m[3]);
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("unknown body in velocity assignment: " + m[1].str());
            continue;
        }

        if (std::regex_match(line, m, orbit_re)) {
            const double start = std::stod(m[2]);
            const double stop = std::stod(m[3]);
            p.steps = std::max(1, static_cast<int>(std::round(stop - start)));
            p.dt = std::stod(m[5]) * unit_scale(m[6]);
            std::regex integrator_re(R"(integrator\s+([A-Za-z_][A-Za-z0-9_]*))");
            std::smatch integrator_match;
            if (std::regex_search(line, integrator_match, integrator_re)) p.integrator = integrator_match[1];
            in_block = true;
            has_sim = true;
            continue;
        }

        if (std::regex_match(line, m, friction_re)) {
            p.friction = std::stod(m[1]);
            continue;
        }

        if (line == "monitor energy") {
            p.monitor_energy = true;
            continue;
        }

        if (line == "monitor momentum") {
            p.monitor_momentum = true;
            continue;
        }

        if (line == "monitor angular_momentum") {
            p.monitor_angular_momentum = true;
            continue;
        }

        if (line == "collisions on") {
            p.collisions_on = true;
            continue;
        }

        if (std::regex_match(line, m, grav_all_re)) {
            for (const auto& s : p.bodies) {
                for (const auto& t : p.bodies) {
                    if (s.name != t.name) p.pulls.push_back({s.name, t.name});
                }
            }
            continue;
        }

        if (std::regex_match(line, m, thrust_re)) {
            p.step_thrusts.push_back({m[1], parse_vec(m[2], m[3])});
            continue;
        }

        if (std::regex_match(line, m, orbital_re)) {
            p.orbital_requests.push_back({m[1], m[2], !has_sim});
            continue;
        }

        if (std::regex_match(line, m, observe_re)) {
            ObserveSpec obs;
            obs.name = m[1];
            obs.kind = (m[2] == "velocity") ? ObserveKind::Velocity : ObserveKind::Position;
            obs.output_file = m[3];
            obs.frequency = std::max(1, std::stoi(m[4]));
            p.observe_specs.push_back(obs);
            continue;
        }

        if (std::regex_match(line, m, step_physics_re)) {
            p.pulls.push_back({m[2], m[1]});
            continue;
        }

        if (std::regex_match(line, m, adaptive_re)) {
            p.adaptive_on = true;
            p.adaptive_tol = std::stod(m[1]);
            p.adaptive_dt_min = std::stod(m[2]) * unit_scale(m[3]);
            p.adaptive_dt_max = std::stod(m[4]) * unit_scale(m[5]);
            if (p.adaptive_dt_min <= 0.0 || p.adaptive_dt_max <= 0.0 || p.adaptive_dt_min > p.adaptive_dt_max) {
                throw std::runtime_error("invalid adaptive timestep bounds");
            }
            continue;
        }

        if (std::regex_match(line, m, softening_re)) {
            p.softening = std::max(0.0, std::stod(m[1]) * unit_scale(m[2]));
            continue;
        }

        if (std::regex_match(line, m, gravity_constant_re)) {
            p.gravity_constant = std::stod(m[1]);
            continue;
        }

        if (std::regex_match(line, m, threads_re)) {
            const std::string threads_value = m[1];
            if (threads_value == "auto") {
                p.worker_threads = 0;
            } else {
                p.worker_threads = std::max(1, std::stoi(threads_value));
            }
            continue;
        }

        if (std::regex_match(line, m, threading_min_re)) {
            p.threading_min_interactions = std::max<size_t>(1, static_cast<size_t>(std::stoull(m[1])));
            continue;
        }

        if (std::regex_match(line, m, profile_re)) {
            p.profile_on = (m[1] == "on");
            continue;
        }

        if (std::regex_match(line, m, gravity_model_re)) {
            const std::string model = m[1];
            if (model == "newtonian") {
                p.gravity_model = GravityModel::Newtonian;
            } else if (model == "mond") {
                p.gravity_model = GravityModel::Mond;
                if (m[2].matched) p.mond_a0 = std::stod(m[2]);
            } else {
                p.gravity_model = GravityModel::GRCorrection;
                if (m[4].matched) p.gr_beta = std::stod(m[4]);
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

        if (std::regex_match(line, m, print_re)) {
            if (m[2] == "position") p.print_position_names.push_back(m[1]);
            else p.print_velocity_names.push_back(m[1]);
            continue;
        }

        if (strict_mode) throw std::runtime_error("unsupported line " + std::to_string(line_no) + ": " + line);
        std::cerr << "warning: ignored unsupported line " << line_no << ": " << line << "\n";
    }

    if (p.bodies.empty()) throw std::runtime_error("no sphere/probe objects found");
    if (has_sim && p.steps <= 0) throw std::runtime_error("no simulate/orbit loop found");
    if (has_sim && p.pulls.empty() && p.bodies.size() > 1) {
        for (const auto& s : p.bodies) {
            for (const auto& t : p.bodies) {
                if (s.name != t.name) p.pulls.push_back({s.name, t.name});
            }
        }
    }
    return p;
}

unsigned int determine_worker_threads(unsigned int requested_threads, size_t interaction_count, size_t min_interactions_for_threads) {
    if (interaction_count < min_interactions_for_threads) return 1;

    unsigned int workers = requested_threads;
    if (workers == 0) {
        if (const char* env = std::getenv("GRAVITY_THREADS")) {
            try {
                const unsigned long parsed = std::stoul(env);
                workers = static_cast<unsigned int>(std::max<unsigned long>(1UL, parsed));
            } catch (const std::exception&) {
                workers = 0;
            }
        }
    }

    if (workers == 0) workers = std::thread::hardware_concurrency();
    if (workers == 0) workers = 1;
    return std::min<unsigned int>(workers, static_cast<unsigned int>(interaction_count));
}


struct AccelScratch {
    std::vector<std::vector<long double>> ax_local;
    std::vector<std::vector<long double>> ay_local;
    std::vector<std::vector<long double>> az_local;

    void ensure(unsigned int worker_count, size_t body_count) {
        if (worker_count <= 1) return;
        if (ax_local.size() != worker_count || (!ax_local.empty() && ax_local[0].size() != body_count)) {
            ax_local.assign(worker_count, std::vector<long double>(body_count, 0.0L));
            ay_local.assign(worker_count, std::vector<long double>(body_count, 0.0L));
            az_local.assign(worker_count, std::vector<long double>(body_count, 0.0L));
            return;
        }
        for (unsigned int w = 0; w < worker_count; ++w) {
            std::fill(ax_local[w].begin(), ax_local[w].end(), 0.0L);
            std::fill(ay_local[w].begin(), ay_local[w].end(), 0.0L);
            std::fill(az_local[w].begin(), az_local[w].end(), 0.0L);
        }
    }
};

std::vector<Vec3> compute_acc(const std::vector<Body>& bodies,
                             const std::vector<std::pair<int, int>>& pulls,
                             double softening,
                             GravityModel gravity_model,
                             double mond_a0,
                             double gr_beta,
                             double gravity_constant,
                             unsigned int resolved_threads,
                             AccelScratch* scratch) {
    const long double G = static_cast<long double>(gravity_constant);
    std::vector<Vec3> a(bodies.size(), {0.0, 0.0, 0.0});
    std::vector<long double> ax(bodies.size(), 0.0L), ay(bodies.size(), 0.0L), az(bodies.size(), 0.0L);
    const long double s2 = static_cast<long double>(softening) * static_cast<long double>(softening);

    const unsigned int worker_count = std::max(1u, resolved_threads);
    if (worker_count == 1) {
        for (const auto& pair : pulls) {
            const int s = pair.first;
            const int t = pair.second;
            const auto& src = bodies[static_cast<size_t>(s)];
            const auto& dst = bodies[static_cast<size_t>(t)];
            const long double dx = static_cast<long double>(src.pos[0]) - static_cast<long double>(dst.pos[0]);
            const long double dy = static_cast<long double>(src.pos[1]) - static_cast<long double>(dst.pos[1]);
            const long double dz = static_cast<long double>(src.pos[2]) - static_cast<long double>(dst.pos[2]);
            const long double r2 = dx * dx + dy * dy + dz * dz + s2 + 1e-18L;
            const long double r = std::sqrt(r2);
            long double am = G * static_cast<long double>(src.mass) / r2;
            if (gravity_model == GravityModel::Mond) {
                const long double aN = am;
                const long double a0 = static_cast<long double>(mond_a0);
                am = std::sqrt(aN * aN + aN * a0);
            } else if (gravity_model == GravityModel::GRCorrection) {
                const long double beta = static_cast<long double>(gr_beta);
                am *= (1.0L + beta / std::max(1e-18L, r2));
            }
            ax[static_cast<size_t>(t)] += am * dx / r;
            ay[static_cast<size_t>(t)] += am * dy / r;
            az[static_cast<size_t>(t)] += am * dz / r;
        }
    } else {
        if (scratch == nullptr) throw std::runtime_error("internal error: missing acceleration scratch buffer");
        scratch->ensure(worker_count, bodies.size());
        auto& ax_local = scratch->ax_local;
        auto& ay_local = scratch->ay_local;
        auto& az_local = scratch->az_local;
        std::vector<std::thread> workers;
        workers.reserve(worker_count);

        for (unsigned int w = 0; w < worker_count; ++w) {
            workers.emplace_back([&, w]() {
                const size_t begin = pulls.size() * static_cast<size_t>(w) / worker_count;
                const size_t end = pulls.size() * static_cast<size_t>(w + 1) / worker_count;
                auto& axw = ax_local[w];
                auto& ayw = ay_local[w];
                auto& azw = az_local[w];
                for (size_t i = begin; i < end; ++i) {
                    const int s = pulls[i].first;
                    const int t = pulls[i].second;
                    const auto& src = bodies[static_cast<size_t>(s)];
                    const auto& dst = bodies[static_cast<size_t>(t)];
                    const long double dx = static_cast<long double>(src.pos[0]) - static_cast<long double>(dst.pos[0]);
                    const long double dy = static_cast<long double>(src.pos[1]) - static_cast<long double>(dst.pos[1]);
                    const long double dz = static_cast<long double>(src.pos[2]) - static_cast<long double>(dst.pos[2]);
                    const long double r2 = dx * dx + dy * dy + dz * dz + s2 + 1e-18L;
                    const long double r = std::sqrt(r2);
                    long double am = G * static_cast<long double>(src.mass) / r2;
                    if (gravity_model == GravityModel::Mond) {
                        const long double aN = am;
                        const long double a0 = static_cast<long double>(mond_a0);
                        am = std::sqrt(aN * aN + aN * a0);
                    } else if (gravity_model == GravityModel::GRCorrection) {
                        const long double beta = static_cast<long double>(gr_beta);
                        am *= (1.0L + beta / std::max(1e-18L, r2));
                    }
                    axw[static_cast<size_t>(t)] += am * dx / r;
                    ayw[static_cast<size_t>(t)] += am * dy / r;
                    azw[static_cast<size_t>(t)] += am * dz / r;
                }
            });
        }
        for (auto& worker : workers) worker.join();

        for (unsigned int w = 0; w < worker_count; ++w) {
            for (size_t i = 0; i < bodies.size(); ++i) {
                ax[i] += ax_local[w][i];
                ay[i] += ay_local[w][i];
                az[i] += az_local[w][i];
            }
        }
    }
    for (size_t i = 0; i < bodies.size(); ++i) {
        a[i][0] = static_cast<double>(ax[i]);
        a[i][1] = static_cast<double>(ay[i]);
        a[i][2] = static_cast<double>(az[i]);
    }
    return a;
}

void apply_friction(std::vector<Body>& bodies, double f) {
    for (auto& b : bodies) {
        if (b.fixed) continue;
        b.vel[0] *= (1.0 - f);
        b.vel[1] *= (1.0 - f);
        b.vel[2] *= (1.0 - f);
    }
}

void run_program(Program& p) {
    using Clock = std::chrono::steady_clock;
    std::unordered_map<std::string, int> index;
    for (size_t i = 0; i < p.bodies.size(); ++i) index[p.bodies[i].name] = static_cast<int>(i);

    std::vector<std::pair<int, int>> pulls;
    pulls.reserve(p.pulls.size());
    for (const auto& [src, dst] : p.pulls) {
        if (!index.contains(src) || !index.contains(dst)) throw std::runtime_error("unknown body in pull relation: " + src + " -> " + dst);
        pulls.push_back({index[src], index[dst]});
    }

    const unsigned int resolved_threads = determine_worker_threads(p.worker_threads, pulls.size(), p.threading_min_interactions);
    AccelScratch accel_scratch;
    if (resolved_threads > 1) accel_scratch.ensure(resolved_threads, p.bodies.size());
    const auto sim_started = Clock::now();

    std::vector<std::ofstream> observe_files;
    observe_files.reserve(p.observe_specs.size());
    for (const auto& obs : p.observe_specs) {
        observe_files.emplace_back(obs.output_file);
        if (!observe_files.back()) throw std::runtime_error("failed to open observe output: " + obs.output_file);
        observe_files.back() << "step,x,y,z\n";
    }

    for (const auto& req : p.orbital_requests) {
        if (!req.before_sim) continue;
        if (!index.contains(req.body) || !index.contains(req.center)) continue;
        print_orbital_elements(p.bodies, index[req.body], index[req.center]);
    }

    for (int step = 0; step < p.steps; ++step) {
        for (const auto& thrust : p.step_thrusts) {
            if (!index.contains(thrust.name)) continue;
            auto& o = p.bodies[static_cast<size_t>(index[thrust.name])];
            if (!o.fixed) {
                o.vel[0] += thrust.delta_v[0];
                o.vel[1] += thrust.delta_v[1];
                o.vel[2] += thrust.delta_v[2];
            }
        }

        double dt_step = p.dt;
        if (p.adaptive_on) {
            const auto a_probe = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            double amax = 0.0;
            for (const auto& av : a_probe) {
                const double am = mag(av);
                if (am > amax) amax = am;
            }
            dt_step = std::sqrt(p.adaptive_tol / std::max(1e-18, amax + 1e-18));
            dt_step = std::clamp(dt_step, p.adaptive_dt_min, p.adaptive_dt_max);
        }

        if (p.integrator == "euler") {
            auto a = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                p.bodies[i].vel[0] += a[i][0] * dt_step;
                p.bodies[i].vel[1] += a[i][1] * dt_step;
                p.bodies[i].vel[2] += a[i][2] * dt_step;
                p.bodies[i].pos[0] += p.bodies[i].vel[0] * dt_step;
                p.bodies[i].pos[1] += p.bodies[i].vel[1] * dt_step;
                p.bodies[i].pos[2] += p.bodies[i].vel[2] * dt_step;
            }
        } else if (p.integrator == "leapfrog") {
            auto a1 = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            std::vector<Vec3> half(p.bodies.size(), {0.0, 0.0, 0.0});
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                half[i][0] = p.bodies[i].vel[0] + a1[i][0] * dt_step * 0.5;
                half[i][1] = p.bodies[i].vel[1] + a1[i][1] * dt_step * 0.5;
                half[i][2] = p.bodies[i].vel[2] + a1[i][2] * dt_step * 0.5;
                p.bodies[i].pos[0] += half[i][0] * dt_step;
                p.bodies[i].pos[1] += half[i][1] * dt_step;
                p.bodies[i].pos[2] += half[i][2] * dt_step;
            }
            auto a2 = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                p.bodies[i].vel[0] = half[i][0] + a2[i][0] * dt_step * 0.5;
                p.bodies[i].vel[1] = half[i][1] + a2[i][1] * dt_step * 0.5;
                p.bodies[i].vel[2] = half[i][2] + a2[i][2] * dt_step * 0.5;
            }
        } else if (p.integrator == "rk4") {
            const auto y0 = p.bodies;
            const auto a1 = compute_acc(y0, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);

            auto y1 = y0;
            for (size_t i = 0; i < y1.size(); ++i) {
                if (y1[i].fixed) continue;
                y1[i].pos[0] += y0[i].vel[0] * dt_step * 0.5;
                y1[i].pos[1] += y0[i].vel[1] * dt_step * 0.5;
                y1[i].pos[2] += y0[i].vel[2] * dt_step * 0.5;
                y1[i].vel[0] += a1[i][0] * dt_step * 0.5;
                y1[i].vel[1] += a1[i][1] * dt_step * 0.5;
                y1[i].vel[2] += a1[i][2] * dt_step * 0.5;
            }

            const auto a2 = compute_acc(y1, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            auto y2 = y0;
            for (size_t i = 0; i < y2.size(); ++i) {
                if (y2[i].fixed) continue;
                y2[i].pos[0] += y1[i].vel[0] * dt_step * 0.5;
                y2[i].pos[1] += y1[i].vel[1] * dt_step * 0.5;
                y2[i].pos[2] += y1[i].vel[2] * dt_step * 0.5;
                y2[i].vel[0] += a2[i][0] * dt_step * 0.5;
                y2[i].vel[1] += a2[i][1] * dt_step * 0.5;
                y2[i].vel[2] += a2[i][2] * dt_step * 0.5;
            }

            const auto a3 = compute_acc(y2, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            auto y3 = y0;
            for (size_t i = 0; i < y3.size(); ++i) {
                if (y3[i].fixed) continue;
                y3[i].pos[0] += y2[i].vel[0] * dt_step;
                y3[i].pos[1] += y2[i].vel[1] * dt_step;
                y3[i].pos[2] += y2[i].vel[2] * dt_step;
                y3[i].vel[0] += a3[i][0] * dt_step;
                y3[i].vel[1] += a3[i][1] * dt_step;
                y3[i].vel[2] += a3[i][2] * dt_step;
            }

            const auto a4 = compute_acc(y3, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                p.bodies[i].pos[0] = y0[i].pos[0] + (dt_step / 6.0) * (y0[i].vel[0] + 2.0 * y1[i].vel[0] + 2.0 * y2[i].vel[0] + y3[i].vel[0]);
                p.bodies[i].pos[1] = y0[i].pos[1] + (dt_step / 6.0) * (y0[i].vel[1] + 2.0 * y1[i].vel[1] + 2.0 * y2[i].vel[1] + y3[i].vel[1]);
                p.bodies[i].pos[2] = y0[i].pos[2] + (dt_step / 6.0) * (y0[i].vel[2] + 2.0 * y1[i].vel[2] + 2.0 * y2[i].vel[2] + y3[i].vel[2]);
                p.bodies[i].vel[0] = y0[i].vel[0] + (dt_step / 6.0) * (a1[i][0] + 2.0 * a2[i][0] + 2.0 * a3[i][0] + a4[i][0]);
                p.bodies[i].vel[1] = y0[i].vel[1] + (dt_step / 6.0) * (a1[i][1] + 2.0 * a2[i][1] + 2.0 * a3[i][1] + a4[i][1]);
                p.bodies[i].vel[2] = y0[i].vel[2] + (dt_step / 6.0) * (a1[i][2] + 2.0 * a2[i][2] + 2.0 * a3[i][2] + a4[i][2]);
            }
        } else {
            auto a1 = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            std::vector<Body> tmp = p.bodies;
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                tmp[i].pos[0] = p.bodies[i].pos[0] + p.bodies[i].vel[0] * dt_step + 0.5 * a1[i][0] * dt_step * dt_step;
                tmp[i].pos[1] = p.bodies[i].pos[1] + p.bodies[i].vel[1] * dt_step + 0.5 * a1[i][1] * dt_step * dt_step;
                tmp[i].pos[2] = p.bodies[i].pos[2] + p.bodies[i].vel[2] * dt_step + 0.5 * a1[i][2] * dt_step * dt_step;
            }
            auto a2 = compute_acc(tmp, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch);
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                p.bodies[i].pos = tmp[i].pos;
                p.bodies[i].vel[0] += (a1[i][0] + a2[i][0]) * dt_step * 0.5;
                p.bodies[i].vel[1] += (a1[i][1] + a2[i][1]) * dt_step * 0.5;
                p.bodies[i].vel[2] += (a1[i][2] + a2[i][2]) * dt_step * 0.5;
            }
        }

        apply_friction(p.bodies, p.friction);
        if (p.collisions_on) apply_collisions(p.bodies);

        for (const auto& name : p.print_position_names) {
            if (!index.contains(name)) continue;
            const auto& o = p.bodies[static_cast<size_t>(index[name])];
            std::cout << name << ".position=(" << o.pos[0] << ", " << o.pos[1] << ", " << o.pos[2] << ")\n";
        }
        for (const auto& name : p.print_velocity_names) {
            if (!index.contains(name)) continue;
            const auto& o = p.bodies[static_cast<size_t>(index[name])];
            std::cout << name << ".velocity=(" << o.vel[0] << ", " << o.vel[1] << ", " << o.vel[2] << ")\n";
        }

        for (size_t i = 0; i < p.observe_specs.size(); ++i) {
            const auto& obs = p.observe_specs[i];
            if (!index.contains(obs.name)) continue;
            if ((step + 1) % obs.frequency == 0) {
                const auto& o = p.bodies[static_cast<size_t>(index[obs.name])];
                const Vec3& v = (obs.kind == ObserveKind::Velocity) ? o.vel : o.pos;
                observe_files[i] << (step + 1) << ',' << v[0] << ',' << v[1] << ',' << v[2] << "\n";
            }
        }

        if (p.monitor_energy) {
            std::cout << "energy.step(" << (step + 1) << ")=" << total_energy(p.bodies) << "\n";
        }
        if (p.monitor_momentum) {
            const auto mom = total_momentum(p.bodies);
            const auto com = center_of_mass(p.bodies);
            std::cout << "momentum.step(" << (step + 1) << ")=(" << mom[0] << ", " << mom[1] << ", " << mom[2] << ")"
                      << " com=(" << com[0] << ", " << com[1] << ", " << com[2] << ")\n";
        }
        if (p.monitor_angular_momentum) {
            const auto ang = total_angular_momentum(p.bodies);
            std::cout << "angular_momentum.step(" << (step + 1) << ")=(" << ang[0] << ", " << ang[1] << ", " << ang[2] << ")\n";
        }
    }

    for (const auto& req : p.orbital_requests) {
        if (req.before_sim) continue;
        if (!index.contains(req.body) || !index.contains(req.center)) continue;
        print_orbital_elements(p.bodies, index[req.body], index[req.center]);
    }

    if (p.profile_on) {
        const auto sim_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - sim_started).count();
        std::cout << "profile.runtime_ms=" << sim_ms
                  << " steps=" << p.steps
                  << " pulls=" << pulls.size()
                  << " threads=" << resolved_threads
                  << " threshold=" << p.threading_min_interactions
                  << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    auto print_help = []() {
        std::cout << R"HELP(__   _ __       
   ____ __________ __   _  ____  / /_ (_) /___  __
  / __ `/ ___/ __ `/ | / / / _ \/ __/ / / __ \/ /
 / /_/ / /  / /_/ /| |/ / /  __/ /_ / / /_/ /_/ / 
 \__, /_/   \__,_/ |___/  \___/\__//_/\____/\__, / 
/____/         ENGINE v3.0 [C++ NATIVE]    /____/  

 » Accuracy: 99.2% (NASA-Ref) | Mode: High-Precision (long double)
 » System: Cross-platform Native C++ | Build: 2026.02.28
 » "For Students, By a Yaka Labs"

usage:
  gravity run <script.gravity> [--profile] [--strict]
  gravity check <script.gravity> [--strict]
  gravity list-features
  gravity --help
  gravity --version
)HELP";
    };
    if (argc < 2) {
        print_help();
        std::cout << "\nTip: use `gravity run <script.gravity>` to execute a simulation.\n";
        return 2;
    }

    const std::string command = argv[1];
    if (command == "--help" || command == "-h" || command == "help") {
        print_help();
        return 0;
    }
    if (command == "--version" || command == "version") {
        std::cout << "gravity ENGINE v3.0 [C++ NATIVE]\n";
        return 0;
    }
    if (command == "list-features") {
        std::cout << "integrators: euler, verlet, leapfrog, rk4\n";
        std::cout << "gravity models: newtonian, mond, gr_correction\n";
        std::cout << "threading: threads auto|N, threading min_interactions N, GRAVITY_THREADS (with reusable buffers)\n";
        std::cout << "diagnostics: monitor energy|momentum|angular_momentum, orbital_elements, profile on|off\n";
        return 0;
    }

    if ((command != "run" && command != "check") || argc < 3) {
        print_help();
        return 2;
    }

    bool force_profile = false;
    bool strict_mode = false;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            force_profile = true;
        } else if (arg == "--strict") {
            strict_mode = true;
        } else {
            std::cerr << "error: unknown option for gravity " << command << ": " << arg << "\n";
            return 2;
        }
    }

    try {
        Program p = parse_gravity(argv[2], strict_mode);
        if (force_profile) p.profile_on = true;
        if (command == "check") {
            std::cout << "ok: parsed script with " << p.bodies.size() << " bodies, " << p.pulls.size() << " pull rules, "
                      << p.steps << " steps\n";
            return 0;
        }
        run_program(p);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
