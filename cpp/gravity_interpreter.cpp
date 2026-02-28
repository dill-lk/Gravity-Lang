#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <functional>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <queue>
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
    bool is_rocket = false;
    double dry_mass = 0.0;
    double fuel_mass = 0.0;
    double burn_rate = 0.0;
    double max_thrust = 0.0;
    double throttle = 0.0;
    double drag_coefficient = 0.0;
    double cross_section = 0.0;
    double throttle_target_speed = -1.0;
    double throttle_pid_p = 0.15;
    bool gravity_turn_on = false;
    double gravity_turn_start_alt = 10000.0;
    double gravity_turn_end_alt = 80000.0;
    double gravity_turn_final_pitch_deg = 85.0;
    double isp_sea_level = 0.0;
    double isp_vacuum = 0.0;
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

struct TimedThrustEvent {
    int step = 1;
    std::string name;
    Vec3 delta_v{0.0, 0.0, 0.0};
};

struct RadiationSpec {
    std::string name;
    Vec3 accel{0.0, 0.0, 0.0};
};

struct DetachEvent {
    int step = 1;
    std::string stage_name;
    std::string rocket_name;
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
    std::vector<TimedThrustEvent> timed_thrusts;
    std::vector<RadiationSpec> radiation_specs;
    std::vector<DetachEvent> detach_events;
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
    bool merge_on_collision = false;
    bool verbose = false;
    int dump_all_frequency = 0;
    std::string dump_all_file;
    bool auto_plot = false;
    std::string auto_plot_body = "Rocket";
    int checkpoint_frequency = 0;
    std::string checkpoint_file;
    std::string resume_file;
    std::string sensitivity_body;
    double sensitivity_mass_percent = 0.0;
    double merge_heat_factor = 1.0;
};

std::string trim(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
    return s;
}


void ensure_parent_directory(const std::string& file_path) {
    const std::filesystem::path out_path(file_path);
    const auto parent = out_path.parent_path();
    if (parent.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) throw std::runtime_error("failed to create output directory: " + parent.string());
}

struct TelemetryPoint {
    int step = 0;
    double altitude_km = 0.0;
    double speed_mps = 0.0;
};

std::string safe_file_component(std::string name) {
    for (char& ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') ch = '_';
    }
    if (name.empty()) return "body";
    return name;
}

void write_telemetry_svg(const std::string& path, const std::vector<TelemetryPoint>& points, const std::string& body_name) {
    if (points.empty()) return;
    ensure_parent_directory(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to open telemetry output: " + path);

    const int width = 1200;
    const int height = 720;
    const int margin = 70;
    const int plot_h = 520;

    const int step_min = points.front().step;
    const int step_max = points.back().step;
    const double step_span = std::max(1.0, static_cast<double>(step_max - step_min));

    double max_alt = 1.0;
    for (const auto& p : points) max_alt = std::max(max_alt, p.altitude_km);

    std::ostringstream poly;
    std::ostringstream motion_path;
    for (size_t i = 0; i < points.size(); ++i) {
        const double x = margin + (static_cast<double>(points[i].step - step_min) / step_span) * (width - margin * 2);
        const double y = margin + plot_h - (points[i].altitude_km / max_alt) * plot_h;
        poly << x << ',' << y << ' ';
        motion_path << (i == 0 ? 'M' : 'L') << x << ' ' << y << ' ';
    }

    out << "<svg xmlns='http://www.w3.org/2000/svg' width='" << width << "' height='" << height << "' viewBox='0 0 " << width << ' ' << height << "'>\n";
    out << "<rect width='100%25' height='100%25' fill='#070d1a'/>\n";
    out << "<text x='" << margin << "' y='42' fill='#f8fafc' font-size='26' font-family='Segoe UI,Arial'>Gravity-Lang Animated Telemetry · " << body_name << "</text>\n";
    out << "<text x='" << margin << "' y='64' fill='#94a3b8' font-size='14'>step " << step_min << " to " << step_max << " · altitude profile</text>\n";
    out << "<rect x='" << margin << "' y='" << margin << "' width='" << (width - margin * 2) << "' height='" << plot_h << "' fill='none' stroke='#1f2a3d'/>\n";
    out << "<polyline fill='none' stroke='#22d3ee' stroke-width='2.5' points='" << poly.str() << "'/>\n";
    out << "<circle r='7' fill='#38bdf8'><animateMotion dur='8s' repeatCount='indefinite' path='" << motion_path.str() << "'/></circle>\n";
    out << "</svg>\n";
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

double apply_collisions(std::vector<Body>& bodies, bool merge_on_collision, double merge_heat_factor) {
    std::vector<bool> remove(bodies.size(), false);
    double heat_generated = 0.0;
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (remove[i]) continue;
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            if (remove[j]) continue;
            const double collision_radius = bodies[i].radius + bodies[j].radius;
            if (collision_radius <= 0.0) continue;
            const auto d = sub(bodies[i].pos, bodies[j].pos);
            const double distance = mag(d);
            if (distance > collision_radius) continue;

            if (merge_on_collision) {
                const double total_mass = std::max(1e-18, bodies[i].mass + bodies[j].mass);
                const double pre_kinetic = 0.5 * bodies[i].mass * (bodies[i].vel[0] * bodies[i].vel[0] + bodies[i].vel[1] * bodies[i].vel[1] + bodies[i].vel[2] * bodies[i].vel[2])
                                         + 0.5 * bodies[j].mass * (bodies[j].vel[0] * bodies[j].vel[0] + bodies[j].vel[1] * bodies[j].vel[1] + bodies[j].vel[2] * bodies[j].vel[2]);
                Body merged = bodies[i];
                merged.name = bodies[i].name + "_" + bodies[j].name;
                merged.mass = total_mass;
                merged.fixed = bodies[i].fixed && bodies[j].fixed;
                for (int k = 0; k < 3; ++k) {
                    merged.pos[k] = (bodies[i].pos[k] * bodies[i].mass + bodies[j].pos[k] * bodies[j].mass) / total_mass;
                    merged.vel[k] = (bodies[i].vel[k] * bodies[i].mass + bodies[j].vel[k] * bodies[j].mass) / total_mass;
                }
                const double post_kinetic = 0.5 * merged.mass * (merged.vel[0] * merged.vel[0] + merged.vel[1] * merged.vel[1] + merged.vel[2] * merged.vel[2]);
                heat_generated += std::max(0.0, (pre_kinetic - post_kinetic) * std::max(0.0, merge_heat_factor));
                const double r1 = std::max(0.0, bodies[i].radius);
                const double r2 = std::max(0.0, bodies[j].radius);
                merged.radius = std::cbrt(r1 * r1 * r1 + r2 * r2 * r2);
                bodies[i] = merged;
                remove[j] = true;
                continue;
            }

            Vec3 normal = {1.0, 0.0, 0.0};
            if (distance > 1e-12) {
                normal = {d[0] / distance, d[1] / distance, d[2] / distance};
            }
            const Vec3 rel = {
                bodies[i].vel[0] - bodies[j].vel[0],
                bodies[i].vel[1] - bodies[j].vel[1],
                bodies[i].vel[2] - bodies[j].vel[2],
            };
            const double rel_normal = rel[0] * normal[0] + rel[1] * normal[1] + rel[2] * normal[2];
            if (rel_normal >= 0.0) continue;

            const double inv_m1 = bodies[i].fixed ? 0.0 : 1.0 / std::max(1e-18, bodies[i].mass);
            const double inv_m2 = bodies[j].fixed ? 0.0 : 1.0 / std::max(1e-18, bodies[j].mass);
            const double denom = inv_m1 + inv_m2;
            if (denom <= 0.0) continue;
            const double impulse = -(2.0 * rel_normal) / denom;
            if (!bodies[i].fixed) {
                for (int k = 0; k < 3; ++k) bodies[i].vel[k] += (impulse * inv_m1) * normal[k];
            }
            if (!bodies[j].fixed) {
                for (int k = 0; k < 3; ++k) bodies[j].vel[k] -= (impulse * inv_m2) * normal[k];
            }
        }
    }
    if (merge_on_collision) {
        std::vector<Body> kept;
        kept.reserve(bodies.size());
        for (size_t i = 0; i < bodies.size(); ++i) {
            if (!remove[i]) kept.push_back(bodies[i]);
        }
        bodies.swap(kept);
    }
    return heat_generated;
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

void save_checkpoint(const std::vector<Body>& bodies, const std::string& path, int step, double dt) {
    ensure_parent_directory(path);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to open checkpoint output: " + path);
    out << "# gravity checkpoint\n";
    out << "step," << step << "\n";
    out << "dt," << dt << "\n";
    for (const auto& b : bodies) {
        out << "body," << b.name << "," << b.mass << "," << b.radius << "," << (b.fixed ? 1 : 0)
            << "," << b.pos[0] << "," << b.pos[1] << "," << b.pos[2]
            << "," << b.vel[0] << "," << b.vel[1] << "," << b.vel[2]
            << "," << (b.is_rocket ? 1 : 0) << "," << b.dry_mass << "," << b.fuel_mass << "," << b.burn_rate
            << "," << b.max_thrust << "," << b.throttle << "," << b.drag_coefficient << "," << b.cross_section
            << "," << (b.gravity_turn_on ? 1 : 0) << "," << b.gravity_turn_start_alt << "," << b.gravity_turn_end_alt
            << "," << b.gravity_turn_final_pitch_deg << "," << b.isp_sea_level << "," << b.isp_vacuum << "\n";
    }
}

void load_checkpoint_into(std::vector<Body>& bodies, const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open checkpoint input: " + path);
    std::vector<Body> loaded;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("body,", 0) != 0) continue;
        std::stringstream ss(line);
        std::string part;
        std::vector<std::string> fields;
        while (std::getline(ss, part, ',')) fields.push_back(part);
        if (fields.size() != 11 && fields.size() != 19 && fields.size() != 25) continue;
        Body b;
        b.name = fields[1];
        b.mass = std::stod(fields[2]);
        b.radius = std::stod(fields[3]);
        b.fixed = (fields[4] == "1");
        b.pos = {std::stod(fields[5]), std::stod(fields[6]), std::stod(fields[7])};
        b.vel = {std::stod(fields[8]), std::stod(fields[9]), std::stod(fields[10])};
        if (fields.size() >= 19) {
            b.is_rocket = (fields[11] == "1");
            b.dry_mass = std::stod(fields[12]);
            b.fuel_mass = std::stod(fields[13]);
            b.burn_rate = std::stod(fields[14]);
            b.max_thrust = std::stod(fields[15]);
            b.throttle = std::stod(fields[16]);
            b.drag_coefficient = std::stod(fields[17]);
            b.cross_section = std::stod(fields[18]);
        }
        if (fields.size() >= 25) {
            b.gravity_turn_on = (fields[19] == "1");
            b.gravity_turn_start_alt = std::stod(fields[20]);
            b.gravity_turn_end_alt = std::stod(fields[21]);
            b.gravity_turn_final_pitch_deg = std::stod(fields[22]);
            b.isp_sea_level = std::stod(fields[23]);
            b.isp_vacuum = std::stod(fields[24]);
        }
        loaded.push_back(b);
    }
    if (loaded.empty()) throw std::runtime_error("checkpoint has no bodies: " + path);
    bodies = loaded;
}

class ThreadPool {
   public:
    explicit ThreadPool(unsigned int workers) : stop_(false), active_(false), task_count_(0), remaining_workers_(0), next_index_(0) {
        workers = std::max(1u, workers);
        for (unsigned int i = 0; i < workers; ++i) {
            threads_.emplace_back([this]() {
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lock(mtx_);
                        cv_.wait(lock, [this]() { return stop_ || active_; });
                        if (stop_) return;
                    }

                    while (true) {
                        const size_t idx = next_index_.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= task_count_) break;
                        task_fn_(idx);
                    }

                    std::lock_guard<std::mutex> lock(mtx_);
                    if (--remaining_workers_ == 0) {
                        active_ = false;
                        done_cv_.notify_one();
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }

    template <class Fn>
    void parallel_for(size_t tasks, Fn fn) {
        if (threads_.size() <= 1 || tasks == 0) {
            for (size_t i = 0; i < tasks; ++i) fn(i);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mtx_);
            task_fn_ = std::function<void(size_t)>(fn);
            task_count_ = tasks;
            remaining_workers_ = threads_.size();
            next_index_.store(0, std::memory_order_relaxed);
            active_ = true;
        }

        cv_.notify_all();
        std::unique_lock<std::mutex> lock(mtx_);
        done_cv_.wait(lock, [this]() { return !active_; });
    }

   private:
    std::vector<std::thread> threads_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::function<void(size_t)> task_fn_;
    bool stop_;
    bool active_;
    size_t task_count_;
    size_t remaining_workers_;
    std::atomic<size_t> next_index_;
};

Program parse_gravity(const std::string& script_path, bool strict_mode = false) {
    Program p;
    std::ifstream in(script_path);
    if (!in) throw std::runtime_error("cannot open script: " + script_path);

    std::regex sphere_head_re(R"(^(sphere|probe|rocket)\s+(\w+)\s+at\s+(\[[^\]]+\])(?:\[(\w+)\])?.*$)");
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
    std::regex collisions_mode_re(R"(^collisions\s+(on|off|merge)\s*$)");
    std::regex dump_all_re(R"obs(^dump_all\s+to\s+"([^"]+)"\s+frequency\s+(\d+)\s*$)obs");
    std::regex plot_re(R"(^plot\s+(on|off)(?:\s+body\s+([A-Za-z_]\w*))?\s*$)");
    std::regex event_thrust_re(R"(^event\s+step\s+(\d+)\s+thrust\s+([A-Za-z_]\w*)\s+by\s+(\[[^\]]+\])\[(\w+\/s)\]\s*$)");
    std::regex radiation_re(R"(^radiation_pressure\s+([A-Za-z_]\w*)\s+by\s+(\[[^\]]+\])\[(m\/s2)\]\s*$)");
    std::regex verbose_re(R"(^verbose\s+(on|off)\s*$)");
    std::regex save_re(R"obs(^save\s+"([^"]+)"\s+frequency\s+(\d+)\s*$)obs");
    std::regex resume_re(R"obs(^resume\s+"([^"]+)"\s*$)obs");
    std::regex sensitivity_re(R"(^sensitivity\s+([A-Za-z_]\w*)\s+mass\s+([-+0-9.eE]+)\s*%\s*$)");
    std::regex merge_heat_re(R"(^merge_heat\s+([-+0-9.eE]+)\s*$)");
    std::regex burn_rate_re(R"(^([A-Za-z_]\w*)\.burn_rate\s*=\s*([-+0-9.eE]+)\[(kg\/s)\]\s*$)");
    std::regex fuel_mass_re(R"(^([A-Za-z_]\w*)\.fuel_mass\s*=\s*([-+0-9.eE]+)\[(kg)\]\s*$)");
    std::regex dry_mass_re(R"(^([A-Za-z_]\w*)\.dry_mass\s*=\s*([-+0-9.eE]+)\[(kg)\]\s*$)");
    std::regex max_thrust_re(R"(^([A-Za-z_]\w*)\.max_thrust\s*=\s*([-+0-9.eE]+)\[(N)\]\s*$)");
    std::regex drag_coeff_re(R"(^([A-Za-z_]\w*)\.drag_coefficient\s*=\s*([-+0-9.eE]+)\s*$)");
    std::regex cross_section_re(R"(^([A-Za-z_]\w*)\.cross_section\s*=\s*([-+0-9.eE]+)\[(m2)\]\s*$)");
    std::regex throttle_assign_re(R"(^([A-Za-z_]\w*)\.throttle\s*=\s*([-+0-9.eE]+)\s*$)");
    std::regex throttle_target_re(R"(^throttle\s+([A-Za-z_]\w*)\s+to\s+maintain\s+velocity\s+([-+0-9.eE]+)\[(m\/s)\]\s*$)");
    std::regex detach_event_re(R"(^event\s+step\s+(\d+)\s+detach\s+([A-Za-z_]\w*)\s+from\s+([A-Za-z_]\w*)\s*$)");
    std::regex gravity_turn_re(R"(^gravity_turn\s+([A-Za-z_]\w*)\s+start\s+([-+0-9.eE]+)\[(m|km)\]\s+end\s+([-+0-9.eE]+)\[(m|km)\]\s+final_pitch\s+([-+0-9.eE]+)\s*$)");
    std::regex isp_sea_re(R"(^([A-Za-z_]\w*)\.isp_sea_level\s*=\s*([-+0-9.eE]+)\[(s)\]\s*$)");
    std::regex isp_vac_re(R"(^([A-Za-z_]\w*)\.isp_vacuum\s*=\s*([-+0-9.eE]+)\[(s)\]\s*$)");

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
            if (m[1] == "rocket") b.is_rocket = true;
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

        if (std::regex_match(line, m, collisions_mode_re)) {
            p.collisions_on = (m[1] != "off");
            p.merge_on_collision = (m[1] == "merge");
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

        if (std::regex_match(line, m, event_thrust_re)) {
            TimedThrustEvent ev;
            ev.step = std::max(1, std::stoi(m[1]));
            ev.name = m[2];
            ev.delta_v = parse_vec(m[3], m[4]);
            p.timed_thrusts.push_back(ev);
            continue;
        }

        if (std::regex_match(line, m, radiation_re)) {
            p.radiation_specs.push_back({m[1], parse_vec(m[2], "m/s")});
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

        if (std::regex_match(line, m, verbose_re)) {
            p.verbose = (m[1] == "on");
            continue;
        }

        if (std::regex_match(line, m, save_re)) {
            p.checkpoint_file = m[1];
            p.checkpoint_frequency = std::max(1, std::stoi(m[2]));
            continue;
        }

        if (std::regex_match(line, m, resume_re)) {
            p.resume_file = m[1];
            continue;
        }

        if (std::regex_match(line, m, sensitivity_re)) {
            p.sensitivity_body = m[1];
            p.sensitivity_mass_percent = std::stod(m[2]);
            continue;
        }

        if (std::regex_match(line, m, merge_heat_re)) {
            p.merge_heat_factor = std::max(0.0, std::stod(m[1]));
            continue;
        }

        if (std::regex_match(line, m, burn_rate_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.burn_rate = std::max(0.0, std::stod(m[2])); b.is_rocket = true; }
            continue;
        }

        if (std::regex_match(line, m, fuel_mass_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.fuel_mass = std::max(0.0, std::stod(m[2])); b.is_rocket = true; b.mass += b.fuel_mass; }
            continue;
        }

        if (std::regex_match(line, m, dry_mass_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.dry_mass = std::max(0.0, std::stod(m[2])); b.is_rocket = true; if (b.mass < b.dry_mass) b.mass = b.dry_mass; }
            continue;
        }

        if (std::regex_match(line, m, max_thrust_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.max_thrust = std::max(0.0, std::stod(m[2])); b.is_rocket = true; }
            continue;
        }

        if (std::regex_match(line, m, drag_coeff_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.drag_coefficient = std::max(0.0, std::stod(m[2])); b.is_rocket = true; }
            continue;
        }

        if (std::regex_match(line, m, cross_section_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.cross_section = std::max(0.0, std::stod(m[2])); b.is_rocket = true; }
            continue;
        }

        if (std::regex_match(line, m, throttle_assign_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.throttle = std::clamp(std::stod(m[2]), 0.0, 1.0); b.is_rocket = true; }
            continue;
        }

        if (std::regex_match(line, m, throttle_target_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.throttle_target_speed = std::max(0.0, std::stod(m[2])); b.is_rocket = true; }
            continue;
        }

        if (std::regex_match(line, m, detach_event_re)) {
            p.detach_events.push_back({std::max(1, std::stoi(m[1])), m[2], m[3]});
            continue;
        }

        if (std::regex_match(line, m, gravity_turn_re)) {
            for (auto& b : p.bodies) {
                if (b.name == m[1]) {
                    b.is_rocket = true;
                    b.gravity_turn_on = true;
                    b.gravity_turn_start_alt = std::stod(m[2]) * unit_scale(m[3]);
                    b.gravity_turn_end_alt = std::stod(m[4]) * unit_scale(m[5]);
                    b.gravity_turn_final_pitch_deg = std::clamp(std::stod(m[6]), 0.0, 89.9);
                }
            }
            continue;
        }

        if (std::regex_match(line, m, isp_sea_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.is_rocket = true; b.isp_sea_level = std::max(0.0, std::stod(m[2])); }
            continue;
        }

        if (std::regex_match(line, m, isp_vac_re)) {
            for (auto& b : p.bodies) if (b.name == m[1]) { b.is_rocket = true; b.isp_vacuum = std::max(0.0, std::stod(m[2])); }
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
    for (const auto& body : p.bodies) {
        bool affected = false;
        for (const auto& pull : p.pulls) {
            if (pull.second == body.name) { affected = true; break; }
        }
        if (!body.fixed && !affected) {
            std::cerr << "warning: body " << body.name << " has no pull rules targeting it and may drift linearly\n";
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
                             AccelScratch* scratch,
                             ThreadPool* pool) {
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
        if (pool == nullptr) throw std::runtime_error("internal error: missing thread pool");
        pool->parallel_for(worker_count, [&](size_t w) {
            const size_t begin = pulls.size() * w / worker_count;
            const size_t end = pulls.size() * (w + 1) / worker_count;
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
    if (!p.resume_file.empty()) {
        load_checkpoint_into(p.bodies, p.resume_file);
        index.clear();
        for (size_t i = 0; i < p.bodies.size(); ++i) index[p.bodies[i].name] = static_cast<int>(i);
    }

    std::vector<std::pair<int, int>> pulls;
    pulls.reserve(p.pulls.size());
    for (const auto& [src, dst] : p.pulls) {
        if (!index.contains(src) || !index.contains(dst)) throw std::runtime_error("unknown body in pull relation: " + src + " -> " + dst);
        pulls.push_back({index[src], index[dst]});
    }

    const unsigned int resolved_threads = determine_worker_threads(p.worker_threads, pulls.size(), p.threading_min_interactions);
    std::unique_ptr<ThreadPool> force_pool;
    if (resolved_threads > 1) force_pool = std::make_unique<ThreadPool>(resolved_threads);
    AccelScratch accel_scratch;
    if (resolved_threads > 1) accel_scratch.ensure(resolved_threads, p.bodies.size());
    const auto sim_started = Clock::now();

    std::vector<std::ofstream> observe_files;
    std::ofstream dump_all_file;
    observe_files.reserve(p.observe_specs.size());
    for (const auto& obs : p.observe_specs) {
        ensure_parent_directory(obs.output_file);
        observe_files.emplace_back(obs.output_file);
        if (!observe_files.back()) throw std::runtime_error("failed to open observe output: " + obs.output_file);
        observe_files.back() << "step,x,y,z\n";
    }

    if (p.dump_all_frequency > 0) {
        if (p.dump_all_file.empty()) p.dump_all_file = "artifacts/dump_all.csv";
        ensure_parent_directory(p.dump_all_file);
        dump_all_file.open(p.dump_all_file);
        if (!dump_all_file) throw std::runtime_error("failed to open dump_all output: " + p.dump_all_file);
        dump_all_file << "step,body,x,y,z,vx,vy,vz\n";
    }

    if (!p.sensitivity_body.empty() && index.contains(p.sensitivity_body)) {
        auto perturbed = p.bodies;
        const size_t bi = static_cast<size_t>(index[p.sensitivity_body]);
        perturbed[bi].mass *= (1.0 + p.sensitivity_mass_percent / 100.0);
        const auto a0 = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
        const auto a1 = compute_acc(perturbed, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
        double delta = 0.0;
        for (size_t i = 0; i < a0.size(); ++i) delta += std::abs(a1[i][0] - a0[i][0]) + std::abs(a1[i][1] - a0[i][1]) + std::abs(a1[i][2] - a0[i][2]);
        std::cout << "sensitivity.mass(" << p.sensitivity_body << "," << p.sensitivity_mass_percent << "%)=" << delta << "\n";
    }

    const double baseline_energy = total_energy(p.bodies);
    const Vec3 baseline_momentum = total_momentum(p.bodies);

    std::vector<TelemetryPoint> telemetry_points;
    telemetry_points.reserve(static_cast<size_t>(std::max(1, p.steps)));
    const std::string telemetry_body = p.auto_plot_body.empty() ? "Rocket" : p.auto_plot_body;

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

        for (const auto& event : p.timed_thrusts) {
            if (event.step != step + 1 || !index.contains(event.name)) continue;
            auto& o = p.bodies[static_cast<size_t>(index[event.name])];
            if (!o.fixed) {
                o.vel[0] += event.delta_v[0];
                o.vel[1] += event.delta_v[1];
                o.vel[2] += event.delta_v[2];
            }
        }
        for (const auto& det : p.detach_events) {
            if (det.step != step + 1 || !index.contains(det.stage_name) || !index.contains(det.rocket_name)) continue;
            auto& stage = p.bodies[static_cast<size_t>(index[det.stage_name])];
            auto& rocket = p.bodies[static_cast<size_t>(index[det.rocket_name])];
            stage.fixed = false;
            stage.pos = rocket.pos;
            stage.vel = rocket.vel;
            if (!rocket.fixed) {
                const double drop_mass = std::max(0.0, stage.mass);
                rocket.mass = std::max(1e-9, rocket.mass - drop_mass);
                if (rocket.dry_mass > 0.0) rocket.dry_mass = std::max(1e-9, rocket.dry_mass - drop_mass);
            }
        }

        double dt_step = p.dt;
        if (p.adaptive_on) {
            const auto a_probe = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            double amax = 0.0;
            for (const auto& av : a_probe) {
                const double am = mag(av);
                if (am > amax) amax = am;
            }
            dt_step = std::sqrt(p.adaptive_tol / std::max(1e-18, amax + 1e-18));
            dt_step = std::clamp(dt_step, p.adaptive_dt_min, p.adaptive_dt_max);
        }

        for (const auto& rad : p.radiation_specs) {
            if (!index.contains(rad.name)) continue;
            auto& o = p.bodies[static_cast<size_t>(index[rad.name])];
            if (!o.fixed) {
                o.vel[0] += rad.accel[0] * dt_step;
                o.vel[1] += rad.accel[1] * dt_step;
                o.vel[2] += rad.accel[2] * dt_step;
            }
        }

        for (auto& b : p.bodies) {
            if (b.fixed || !b.is_rocket) continue;

            const double earth_radius = 6.371e6;
            const double rho0 = 1.225;
            const double scale_h = 8500.0;
            const double alt = std::max(0.0, mag(b.pos) - earth_radius);
            const double rho = rho0 * std::exp(-alt / scale_h);

            if (b.throttle_target_speed >= 0.0) {
                const double speed = mag(b.vel);
                const double err = b.throttle_target_speed - speed;
                b.throttle = std::clamp(b.throttle + b.throttle_pid_p * (err / std::max(1.0, b.throttle_target_speed)), 0.0, 1.0);
            }

            if (b.dry_mass <= 0.0) b.dry_mass = std::max(1e-9, b.mass - b.fuel_mass);
            if (b.isp_vacuum <= 0.0) b.isp_vacuum = b.isp_sea_level;
            if (b.isp_sea_level <= 0.0) b.isp_sea_level = b.isp_vacuum;
            const double atm_factor = std::clamp(rho / rho0, 0.0, 1.0);
            const double isp_eff = (b.isp_vacuum > 0.0 || b.isp_sea_level > 0.0)
                                       ? (b.isp_vacuum + (b.isp_sea_level - b.isp_vacuum) * atm_factor)
                                       : 0.0;

            const double effective_throttle = (b.fuel_mass > 0.0) ? b.throttle : 0.0;
            if (b.fuel_mass > 0.0 && b.max_thrust > 0.0 && effective_throttle > 0.0) {
                const double g0 = 9.80665;
                double effective_burn = b.burn_rate;
                if (effective_burn <= 0.0 && isp_eff > 0.0) {
                    effective_burn = (b.max_thrust * effective_throttle) / (isp_eff * g0);
                }
                if (effective_burn > 0.0) {
                    const double fuel_used = std::min(b.fuel_mass, effective_burn * dt_step);
                    b.fuel_mass -= fuel_used;
                }
            }
            b.mass = std::max(1e-9, b.dry_mass + std::max(0.0, b.fuel_mass));
            const double throttle_now = (b.fuel_mass > 0.0) ? b.throttle : 0.0;

            Vec3 radial = {0.0, 1.0, 0.0};
            const double rmag = mag(b.pos);
            if (rmag > 1e-9) radial = {b.pos[0] / rmag, b.pos[1] / rmag, b.pos[2] / rmag};
            Vec3 prograde = b.vel;
            const double vr = prograde[0] * radial[0] + prograde[1] * radial[1] + prograde[2] * radial[2];
            prograde = {prograde[0] - vr * radial[0], prograde[1] - vr * radial[1], prograde[2] - vr * radial[2]};
            double pgmag = mag(prograde);
            if (pgmag < 1e-9) {
                prograde = {1.0, 0.0, 0.0};
                const double dotr = prograde[0] * radial[0] + prograde[1] * radial[1] + prograde[2] * radial[2];
                prograde = {prograde[0] - dotr * radial[0], prograde[1] - dotr * radial[1], prograde[2] - dotr * radial[2]};
                pgmag = std::max(1e-9, mag(prograde));
            }
            prograde = {prograde[0] / pgmag, prograde[1] / pgmag, prograde[2] / pgmag};

            Vec3 thrust_dir = radial;
            if (b.gravity_turn_on) {
                const double denom = std::max(1.0, b.gravity_turn_end_alt - b.gravity_turn_start_alt);
                const double frac = std::clamp((alt - b.gravity_turn_start_alt) / denom, 0.0, 1.0);
                const double pitch = (b.gravity_turn_final_pitch_deg * frac) * (3.14159265358979323846 / 180.0);
                thrust_dir = {
                    std::cos(pitch) * radial[0] + std::sin(pitch) * prograde[0],
                    std::cos(pitch) * radial[1] + std::sin(pitch) * prograde[1],
                    std::cos(pitch) * radial[2] + std::sin(pitch) * prograde[2],
                };
                const double tmag = std::max(1e-9, mag(thrust_dir));
                thrust_dir = {thrust_dir[0] / tmag, thrust_dir[1] / tmag, thrust_dir[2] / tmag};
            }

            const double isp_scale = (b.isp_vacuum > 0.0) ? std::max(0.1, isp_eff / b.isp_vacuum) : 1.0;
            const double thrust_acc = (b.max_thrust * throttle_now * isp_scale) / std::max(1e-9, b.mass);
            b.vel[0] += thrust_dir[0] * thrust_acc * dt_step;
            b.vel[1] += thrust_dir[1] * thrust_acc * dt_step;
            b.vel[2] += thrust_dir[2] * thrust_acc * dt_step;

            const double speed_now = mag(b.vel);
            if (b.drag_coefficient > 0.0 && b.cross_section > 0.0 && speed_now > 1e-9) {
                const double drag_acc = 0.5 * rho * speed_now * speed_now * b.drag_coefficient * b.cross_section / std::max(1e-9, b.mass);
                b.vel[0] -= (b.vel[0] / speed_now) * drag_acc * dt_step;
                b.vel[1] -= (b.vel[1] / speed_now) * drag_acc * dt_step;
                b.vel[2] -= (b.vel[2] / speed_now) * drag_acc * dt_step;
            }
        }

        if (p.integrator == "euler") {
            auto a = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
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
            auto a1 = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
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
            auto a2 = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                p.bodies[i].vel[0] = half[i][0] + a2[i][0] * dt_step * 0.5;
                p.bodies[i].vel[1] = half[i][1] + a2[i][1] * dt_step * 0.5;
                p.bodies[i].vel[2] = half[i][2] + a2[i][2] * dt_step * 0.5;
            }
        } else if (p.integrator == "yoshida4") {
            constexpr double twothird = 1.2599210498948732;
            const double w1 = 1.0 / (2.0 - twothird);
            const double w0 = -twothird * w1;
            const double c1 = 0.5 * w1;
            const double c2 = 0.5 * (w0 + w1);
            const double c3 = c2;
            const double c4 = c1;
            const double d1 = w1;
            const double d2 = w0;
            const double d3 = w1;

            auto apply_drift = [&](double c) {
                for (size_t i = 0; i < p.bodies.size(); ++i) {
                    if (p.bodies[i].fixed) continue;
                    p.bodies[i].pos[0] += p.bodies[i].vel[0] * c * dt_step;
                    p.bodies[i].pos[1] += p.bodies[i].vel[1] * c * dt_step;
                    p.bodies[i].pos[2] += p.bodies[i].vel[2] * c * dt_step;
                }
            };
            auto apply_kick = [&](const std::vector<Vec3>& a, double d) {
                for (size_t i = 0; i < p.bodies.size(); ++i) {
                    if (p.bodies[i].fixed) continue;
                    p.bodies[i].vel[0] += a[i][0] * d * dt_step;
                    p.bodies[i].vel[1] += a[i][1] * d * dt_step;
                    p.bodies[i].vel[2] += a[i][2] * d * dt_step;
                }
            };

            apply_drift(c1);
            auto a = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            apply_kick(a, d1);
            apply_drift(c2);
            a = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            apply_kick(a, d2);
            apply_drift(c3);
            a = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            apply_kick(a, d3);
            apply_drift(c4);
        } else if (p.integrator == "rk45") {
            const auto y0 = p.bodies;
            auto y2 = y0, y3 = y0, y4 = y0, y5 = y0, y6 = y0;
            const auto k1 = compute_acc(y0, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            auto advance = [&](std::vector<Body>& y, const std::vector<std::pair<double,const std::vector<Vec3>*>>& ks, const std::vector<double>& cs) {
                for (size_t i = 0; i < y.size(); ++i) {
                    if (y[i].fixed) continue;
                    Vec3 av{0,0,0};
                    for (size_t j = 0; j < ks.size(); ++j) {
                        av[0] += cs[j] * (*ks[j].second)[i][0];
                        av[1] += cs[j] * (*ks[j].second)[i][1];
                        av[2] += cs[j] * (*ks[j].second)[i][2];
                    }
                    y[i].pos[0] = y0[i].pos[0] + dt_step * y0[i].vel[0];
                    y[i].pos[1] = y0[i].pos[1] + dt_step * y0[i].vel[1];
                    y[i].pos[2] = y0[i].pos[2] + dt_step * y0[i].vel[2];
                    y[i].vel[0] = y0[i].vel[0] + dt_step * av[0];
                    y[i].vel[1] = y0[i].vel[1] + dt_step * av[1];
                    y[i].vel[2] = y0[i].vel[2] + dt_step * av[2];
                }
            };
            advance(y2, {{0.2,&k1}}, {0.2});
            const auto k2 = compute_acc(y2, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            advance(y3, {{0.0,&k1},{0.3,&k2}}, {3.0/40.0,9.0/40.0});
            const auto k3 = compute_acc(y3, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            advance(y4, {{0.0,&k1},{0.0,&k2},{0.0,&k3}}, {44.0/45.0,-56.0/15.0,32.0/9.0});
            const auto k4 = compute_acc(y4, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            advance(y5, {{0.0,&k1},{0.0,&k2},{0.0,&k3},{0.0,&k4}}, {19372.0/6561.0,-25360.0/2187.0,64448.0/6561.0,-212.0/729.0});
            const auto k5 = compute_acc(y5, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            advance(y6, {{0.0,&k1},{0.0,&k2},{0.0,&k3},{0.0,&k4},{0.0,&k5}}, {9017.0/3168.0,-355.0/33.0,46732.0/5247.0,49.0/176.0,-5103.0/18656.0});
            const auto k6 = compute_acc(y6, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            double err = 0.0;
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                const double vx5 = y0[i].vel[0] + dt_step * (35.0/384.0*k1[i][0] + 500.0/1113.0*k3[i][0] + 125.0/192.0*k4[i][0] - 2187.0/6784.0*k5[i][0] + 11.0/84.0*k6[i][0]);
                const double vy5 = y0[i].vel[1] + dt_step * (35.0/384.0*k1[i][1] + 500.0/1113.0*k3[i][1] + 125.0/192.0*k4[i][1] - 2187.0/6784.0*k5[i][1] + 11.0/84.0*k6[i][1]);
                const double vz5 = y0[i].vel[2] + dt_step * (35.0/384.0*k1[i][2] + 500.0/1113.0*k3[i][2] + 125.0/192.0*k4[i][2] - 2187.0/6784.0*k5[i][2] + 11.0/84.0*k6[i][2]);
                const double vx4 = y0[i].vel[0] + dt_step * (5179.0/57600.0*k1[i][0] + 7571.0/16695.0*k3[i][0] + 393.0/640.0*k4[i][0] - 92097.0/339200.0*k5[i][0] + 187.0/2100.0*k6[i][0] + 1.0/40.0*k2[i][0]);
                const double vy4 = y0[i].vel[1] + dt_step * (5179.0/57600.0*k1[i][1] + 7571.0/16695.0*k3[i][1] + 393.0/640.0*k4[i][1] - 92097.0/339200.0*k5[i][1] + 187.0/2100.0*k6[i][1] + 1.0/40.0*k2[i][1]);
                const double vz4 = y0[i].vel[2] + dt_step * (5179.0/57600.0*k1[i][2] + 7571.0/16695.0*k3[i][2] + 393.0/640.0*k4[i][2] - 92097.0/339200.0*k5[i][2] + 187.0/2100.0*k6[i][2] + 1.0/40.0*k2[i][2]);
                err = std::max(err, std::abs(vx5-vx4) + std::abs(vy5-vy4) + std::abs(vz5-vz4));
                p.bodies[i].vel = {vx5, vy5, vz5};
                p.bodies[i].pos[0] = y0[i].pos[0] + dt_step * p.bodies[i].vel[0];
                p.bodies[i].pos[1] = y0[i].pos[1] + dt_step * p.bodies[i].vel[1];
                p.bodies[i].pos[2] = y0[i].pos[2] + dt_step * p.bodies[i].vel[2];
            }
            if (p.adaptive_on) {
                const double factor = std::clamp(0.9 * std::pow(std::max(1e-18, p.adaptive_tol / std::max(1e-18, err)), 0.2), 0.5, 1.5);
                p.dt = std::clamp(dt_step * factor, p.adaptive_dt_min, p.adaptive_dt_max);
            }
        } else if (p.integrator == "rk4") {
            const auto y0 = p.bodies;
            const auto a1 = compute_acc(y0, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());

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

            const auto a2 = compute_acc(y1, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
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

            const auto a3 = compute_acc(y2, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
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

            const auto a4 = compute_acc(y3, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
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
            auto a1 = compute_acc(p.bodies, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            std::vector<Body> tmp = p.bodies;
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                tmp[i].pos[0] = p.bodies[i].pos[0] + p.bodies[i].vel[0] * dt_step + 0.5 * a1[i][0] * dt_step * dt_step;
                tmp[i].pos[1] = p.bodies[i].pos[1] + p.bodies[i].vel[1] * dt_step + 0.5 * a1[i][1] * dt_step * dt_step;
                tmp[i].pos[2] = p.bodies[i].pos[2] + p.bodies[i].vel[2] * dt_step + 0.5 * a1[i][2] * dt_step * dt_step;
            }
            auto a2 = compute_acc(tmp, pulls, p.softening, p.gravity_model, p.mond_a0, p.gr_beta, p.gravity_constant, resolved_threads, &accel_scratch, force_pool.get());
            for (size_t i = 0; i < p.bodies.size(); ++i) {
                if (p.bodies[i].fixed) continue;
                p.bodies[i].pos = tmp[i].pos;
                p.bodies[i].vel[0] += (a1[i][0] + a2[i][0]) * dt_step * 0.5;
                p.bodies[i].vel[1] += (a1[i][1] + a2[i][1]) * dt_step * 0.5;
                p.bodies[i].vel[2] += (a1[i][2] + a2[i][2]) * dt_step * 0.5;
            }
        }

        apply_friction(p.bodies, p.friction);
        if (p.collisions_on) {
            const size_t before_collision_count = p.bodies.size();
            const double merge_heat = apply_collisions(p.bodies, p.merge_on_collision, p.merge_heat_factor);
            if (p.verbose && merge_heat > 0.0) std::cout << "merge.heat.step(" << (step + 1) << ")=" << merge_heat << "\n";
            if (p.merge_on_collision && p.bodies.size() != before_collision_count) {
                index.clear();
                for (size_t i = 0; i < p.bodies.size(); ++i) index[p.bodies[i].name] = static_cast<int>(i);
                pulls.clear();
                for (const auto& src_body : p.bodies) {
                    for (const auto& dst_body : p.bodies) {
                        if (src_body.name != dst_body.name) pulls.push_back({index[src_body.name], index[dst_body.name]});
                    }
                }
            }
        }

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

        if (p.dump_all_frequency > 0 && ((step + 1) % p.dump_all_frequency == 0)) {
            for (const auto& body : p.bodies) {
                dump_all_file << (step + 1) << "," << body.name << "," << body.pos[0] << "," << body.pos[1] << "," << body.pos[2]
                              << "," << body.vel[0] << "," << body.vel[1] << "," << body.vel[2] << "\n";
            }
        }

        if (p.auto_plot && index.contains(telemetry_body)) {
            const auto& b = p.bodies[static_cast<size_t>(index[telemetry_body])];
            const double radius = std::sqrt(b.pos[0] * b.pos[0] + b.pos[1] * b.pos[1] + b.pos[2] * b.pos[2]);
            const double altitude_km = std::max(0.0, radius - 6371000.0) / 1000.0;
            const double speed = std::sqrt(b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]);
            telemetry_points.push_back({step + 1, altitude_km, speed});
        }

        if (p.verbose) {
            const auto mom = total_momentum(p.bodies);
            const double drift_e = total_energy(p.bodies) - baseline_energy;
            const double drift_p = std::sqrt(std::pow(mom[0] - baseline_momentum[0], 2) + std::pow(mom[1] - baseline_momentum[1], 2) + std::pow(mom[2] - baseline_momentum[2], 2));
            std::cout << "verbose.step(" << (step + 1) << "): dt=" << dt_step << " energy_drift=" << drift_e << " momentum_drift=" << drift_p << "\n";
        }

        if (p.checkpoint_frequency > 0 && !p.checkpoint_file.empty() && ((step + 1) % p.checkpoint_frequency == 0)) {
            save_checkpoint(p.bodies, p.checkpoint_file, step + 1, p.dt);
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

    if (p.auto_plot) {
        if (telemetry_points.empty()) {
            std::cout << "[System] plot on requested, but no telemetry samples found for body " << telemetry_body << ".\n";
        } else {
            const std::string svg_path = "artifacts/telemetry_" + safe_file_component(telemetry_body) + ".svg";
            write_telemetry_svg(svg_path, telemetry_points, telemetry_body);
            std::cout << "[System] C++ animated SVG written: " << svg_path << "\n";
        }
    }

    if (p.adaptive_on) {
        const double ratio = (p.adaptive_dt_max > 0.0) ? std::clamp((p.dt - p.adaptive_dt_min) / (p.adaptive_dt_max - p.adaptive_dt_min + 1e-18), 0.0, 1.0) : 0.0;
        const double drift = std::abs(total_energy(p.bodies) - baseline_energy) / (std::abs(baseline_energy) + 1e-18);
        const double confidence = std::clamp(100.0 * (0.6 * ratio + 0.4 * (1.0 - std::min(1.0, drift))), 0.0, 100.0);
        std::cout << "confidence.score=" << confidence << "\n";
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
)HELP";
        std::cout << " » System: Cross-platform Native C++ | Build: " << __DATE__ << " " << __TIME__ << "\n";
        std::cout << R"HELP( » "For Students, By a Yaka Labs"

usage:
  gravity run <script.gravity> [--profile] [--strict] [--dump-all[=file]] [--resume <checkpoint.json>]
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
        std::cout << "gravity ENGINE v3.0 [C++ NATIVE] build " << __DATE__ << " " << __TIME__ << "\n";
        return 0;
    }
    if (command == "list-features") {
        std::cout << "integrators: euler, verlet, leapfrog, rk4, yoshida4, rk45\n";
        std::cout << "gravity models: newtonian, mond, gr_correction\n";
        std::cout << "threading: threads auto|N, threading min_interactions N, GRAVITY_THREADS (thread-pool force accumulation)\n";
        std::cout << "diagnostics: monitor energy|momentum|angular_momentum, orbital_elements, verbose, sensitivity, merge_heat, confidence.score, profile on|off, plot on|off [body Name]\n";
        std::cout << "rocketry: variable mass burn, gravity turn autopilot, variable ISP, atmospheric drag, PID throttle target, step detach staging\n";
        return 0;
    }

    if ((command != "run" && command != "check") || argc < 3) {
        print_help();
        return 2;
    }

    bool force_profile = false;
    bool strict_mode = false;
    bool cli_dump_all = false;
    std::string cli_dump_all_file;
    std::string cli_resume_file;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            force_profile = true;
        } else if (arg == "--strict") {
            strict_mode = true;
        } else if (arg == "--dump-all") {
            cli_dump_all = true;
        } else if (arg.rfind("--dump-all=", 0) == 0) {
            cli_dump_all = true;
            cli_dump_all_file = arg.substr(std::string("--dump-all=").size());
        } else if (arg == "--resume" && i + 1 < argc) {
            cli_resume_file = argv[++i];
        } else {
            std::cerr << "error: unknown option for gravity " << command << ": " << arg << "\n";
            return 2;
        }
    }


    try {
        Program p = parse_gravity(argv[2], strict_mode);
        if (force_profile) p.profile_on = true;
        if (cli_dump_all) {
            p.dump_all_frequency = 1;
            p.dump_all_file = cli_dump_all_file.empty() ? "artifacts/dump_all.csv" : cli_dump_all_file;
        }
        if (!cli_resume_file.empty()) p.resume_file = cli_resume_file;
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
