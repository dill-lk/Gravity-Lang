#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void compute_acc(
    int n,
    int pair_count,
    const int* src_idx,
    const int* dst_idx,
    const double* masses,
    const double* pos,
    double* acc,
    double softening
) {
    const double G = 6.67430e-11;

    for (int i = 0; i < n * 3; i++) {
        acc[i] = 0.0;
    }

    // Parallel accumulation with thread-local buffers to avoid atomics.
    // This is especially useful for large pair lists.
#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
    double* thread_acc = new double[static_cast<size_t>(thread_count) * n * 3]();

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* local = thread_acc + static_cast<size_t>(tid) * n * 3;

#pragma omp for schedule(static)
        for (int p = 0; p < pair_count; p++) {
            const int s = src_idx[p];
            const int t = dst_idx[p];
            const double dx = pos[s * 3 + 0] - pos[t * 3 + 0];
            const double dy = pos[s * 3 + 1] - pos[t * 3 + 1];
            const double dz = pos[s * 3 + 2] - pos[t * 3 + 2];
            const double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
            const double r = std::sqrt(r2);
            const double invr = 1.0 / r;
            const double a = G * masses[p] / r2;

            local[t * 3 + 0] += a * dx * invr;
            local[t * 3 + 1] += a * dy * invr;
            local[t * 3 + 2] += a * dz * invr;
        }
    }

    for (int t = 0; t < thread_count; t++) {
        const double* local = thread_acc + static_cast<size_t>(t) * n * 3;
        for (int i = 0; i < n * 3; i++) {
            acc[i] += local[i];
        }
    }

    delete[] thread_acc;
#else
    for (int p = 0; p < pair_count; p++) {
        const int s = src_idx[p];
        const int t = dst_idx[p];
        const double dx = pos[s * 3 + 0] - pos[t * 3 + 0];
        const double dy = pos[s * 3 + 1] - pos[t * 3 + 1];
        const double dz = pos[s * 3 + 2] - pos[t * 3 + 2];
        const double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
        const double r = std::sqrt(r2);
        const double invr = 1.0 / r;
        const double a = G * masses[p] / r2;

        acc[t * 3 + 0] += a * dx * invr;
        acc[t * 3 + 1] += a * dy * invr;
        acc[t * 3 + 2] += a * dz * invr;
    }
#endif
}
