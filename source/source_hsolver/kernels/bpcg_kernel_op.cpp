#include "source_hsolver/kernels/bpcg_kernel_op.h"
#include "source_base/module_external/blas_connector.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/parallel_reduce.h"
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace hsolver
{

    namespace
    {
        constexpr int kBpcgOpenmpMinWork = 4096;

        inline bool use_bpcg_openmp(int n)
        {
#ifdef _OPENMP
            return n >= kBpcgOpenmpMinWork && omp_get_max_threads() > 1;
#else
            return false;
#endif
        }
    } // namespace

    template <typename T>
    struct line_minimize_with_block_op<T, base_device::DEVICE_CPU>
    {
        using Real = typename GetTypeReal<T>::type;
        void operator()(T* grad_out,
            T* hgrad_out,
            T* psi_out,
            T* hpsi_out,
            const int& n_basis,
            const int& n_basis_max,
            const int& n_band)
        {
            const bool use_openmp = use_bpcg_openmp(n_basis * n_band);
            std::vector<Real> norm(n_band, 0.0);
            std::vector<Real> epsilo_0(n_band, 0.0);
            std::vector<Real> epsilo_1(n_band, 0.0);
            std::vector<Real> epsilo_2(n_band, 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                auto A = reinterpret_cast<const Real*>(grad_out + band_idx * n_basis_max);
                norm[band_idx] = BlasConnector::dot(2 * n_basis, A, 1, A, 1);
            }
            Parallel_Reduce::reduce_pool(norm.data(), n_band);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                const Real band_norm = 1.0 / sqrt(norm[band_idx]);
                for (int basis_idx = 0; basis_idx < n_basis; basis_idx++)
                {
                    auto item = band_idx * n_basis_max + basis_idx;
                    grad_out[item] *= band_norm;
                    hgrad_out[item] *= band_norm;
                    epsilo_0[band_idx] += std::real(hpsi_out[item] * std::conj(psi_out[item]));
                    epsilo_1[band_idx] += std::real(grad_out[item] * std::conj(hpsi_out[item]));
                    epsilo_2[band_idx] += std::real(grad_out[item] * std::conj(hgrad_out[item]));
                }
            }
            Parallel_Reduce::reduce_pool(epsilo_0.data(), n_band);
            Parallel_Reduce::reduce_pool(epsilo_1.data(), n_band);
            Parallel_Reduce::reduce_pool(epsilo_2.data(), n_band);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                Real theta = 0.5 * std::abs(std::atan(2 * epsilo_1[band_idx] / (epsilo_0[band_idx] - epsilo_2[band_idx])));
                Real cos_theta = std::cos(theta);
                Real sin_theta = std::sin(theta);
                for (int basis_idx = 0; basis_idx < n_basis; basis_idx++)
                {
                    auto item = band_idx * n_basis_max + basis_idx;
                    psi_out[item] = psi_out[item] * cos_theta + grad_out[item] * sin_theta;
                    hpsi_out[item] = hpsi_out[item] * cos_theta + hgrad_out[item] * sin_theta;
                }
            }
        }
    };

    template <typename T>
    struct calc_grad_with_block_op<T, base_device::DEVICE_CPU>
    {
        using Real = typename GetTypeReal<T>::type;
        void operator()(const Real* prec_in,
            Real* err_out,
            Real* beta_out,
            T* psi_out,
            T* hpsi_out,
            T* grad_out,
            T* grad_old_out,
            const int& n_basis,
            const int& n_basis_max,
            const int& n_band)
        {
            const bool use_openmp = use_bpcg_openmp(n_basis * n_band);
            std::vector<Real> norm(n_band, 0.0);
            std::vector<Real> epsilo(n_band, 0.0);
            std::vector<Real> err(n_band, 0.0);
            std::vector<Real> beta(n_band, 0.0);
            std::vector<Real> beta_old(n_band, 0.0);
            std::vector<Real> beta_ratio(n_band, 0.0);

            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                beta_old[band_idx] = beta_out[band_idx];
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                auto A = reinterpret_cast<const Real*>(psi_out + band_idx * n_basis_max);
                norm[band_idx] = BlasConnector::dot(2 * n_basis, A, 1, A, 1);
            }
            Parallel_Reduce::reduce_pool(norm.data(), n_band);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                const Real band_norm = 1.0 / sqrt(norm[band_idx]);
                for (int basis_idx = 0; basis_idx < n_basis; basis_idx++)
                {
                    auto item = band_idx * n_basis_max + basis_idx;
                    psi_out[item] *= band_norm;
                    hpsi_out[item] *= band_norm;
                    epsilo[band_idx] += std::real(hpsi_out[item] * std::conj(psi_out[item]));
                }
            }
            Parallel_Reduce::reduce_pool(epsilo.data(), n_band);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                const Real band_epsilo = epsilo[band_idx];
                for (int basis_idx = 0; basis_idx < n_basis; basis_idx++)
                {
                    auto item = band_idx * n_basis_max + basis_idx;
                    grad_out[item] = hpsi_out[item] - band_epsilo * psi_out[item];
                    const Real grad_2 = std::norm(grad_out[item]);
                    err[band_idx] += grad_2;
                    beta[band_idx] += grad_2 / prec_in[basis_idx]; /// Mark here as we should div the prec?
                }
            }
            Parallel_Reduce::reduce_pool(err.data(), n_band);
            Parallel_Reduce::reduce_pool(beta.data(), n_band);

            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                beta_ratio[band_idx] = beta[band_idx] / beta_old[band_idx];
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
            for (int band_idx = 0; band_idx < n_band; band_idx++)
            {
                const Real band_beta_ratio = beta_ratio[band_idx];
                for (int basis_idx = 0; basis_idx < n_basis; basis_idx++)
                {
                    auto item = band_idx * n_basis_max + basis_idx;
                    grad_out[item] = -grad_out[item] / prec_in[basis_idx] + band_beta_ratio * grad_old_out[item];
                }
                beta_out[band_idx] = beta[band_idx];
                err_out[band_idx] = sqrt(err[band_idx]);
            }
        }
    };

    template <typename T>
    struct apply_eigenvalues_op<T, base_device::DEVICE_CPU>
    {
        using Real = typename GetTypeReal<T>::type;
        void operator()(const int& nbase, const int& nbase_x, const int& notconv, T* result, const T* vectors, const Real* eigenvalues)
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(use_bpcg_openmp(nbase * notconv))
#endif
            for (int m = 0; m < notconv; m++)
            {
                for (int idx = 0; idx < nbase; idx++)
                {
                    result[m * nbase_x + idx] = eigenvalues[m] * vectors[m * nbase_x + idx];
                }
            }
        }
    };

    template <typename T>
    struct precondition_op<T, base_device::DEVICE_CPU> {
        using Real = typename GetTypeReal<T>::type;
        void operator()(const int& dim,
            T* psi_iter,
            const int& nbase,
            const int& notconv,
            const Real* precondition,
            const Real* eigenvalues)
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(use_bpcg_openmp(dim * notconv))
#endif
            for (int m = 0; m < notconv; m++)
            {
                for (int i = 0; i < dim; i++)
                {
                    Real x = std::abs(precondition[i] - eigenvalues[m]);
                    Real denom = 0.5 * (1.0 + x + sqrt(1 + (x - 1.0) * (x - 1.0)));
                    psi_iter[(nbase + m) * dim + i] /= denom;
                }
            }
        }
    };

    template <typename T>
    struct normalize_op<T, base_device::DEVICE_CPU> {
        void operator()(const int& dim,
            T* psi_iter,
            const int& nbase,
            const int& notconv,
            typename GetTypeReal<T>::type* psi_norm)
        {
            using Real = typename GetTypeReal<T>::type;
            for (int m = 0; m < notconv; m++)
            {
                // Calculate norm using dot_real_op
                Real psi_m_norm = ModuleBase::dot_real_op<T, base_device::DEVICE_CPU>()(
                    dim,
                    psi_iter + (nbase + m) * dim,
                    psi_iter + (nbase + m) * dim,
                    true);
                assert(psi_m_norm > 0.0);
                psi_m_norm = sqrt(psi_m_norm);

                // Normalize using vector_div_constant_op
                ModuleBase::vector_div_constant_op<T, base_device::DEVICE_CPU>()(
                    dim,
                    psi_iter + (nbase + m) * dim,
                    psi_iter + (nbase + m) * dim,
                    psi_m_norm);
                if (psi_norm) {
                    psi_norm[m] = psi_m_norm;
                }
            }
        }
    };

    template <typename T>
    struct refresh_hcc_scc_vcc_op<T, base_device::DEVICE_CPU>
    {
        using Real = typename GetTypeReal<T>::type;
        void operator()(const int& n,
            T* hcc,
            T* scc,
            T* vcc,
            const int& ldh,
            const Real* eigenvalue,
            const T& one)
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(1) schedule(static)
#endif
            for (int i = 0; i < n; i++)
            {
                hcc[i * ldh + i] = eigenvalue[i];
                scc[i * ldh + i] = one;
                vcc[i * ldh + i] = one;
            }
        }
    };

    template struct calc_grad_with_block_op<std::complex<float>, base_device::DEVICE_CPU>;
    template struct line_minimize_with_block_op<std::complex<float>, base_device::DEVICE_CPU>;
    template struct calc_grad_with_block_op<std::complex<double>, base_device::DEVICE_CPU>;
    template struct line_minimize_with_block_op<std::complex<double>, base_device::DEVICE_CPU>;
    template struct apply_eigenvalues_op<std::complex<float>, base_device::DEVICE_CPU>;
    template struct apply_eigenvalues_op<std::complex<double>, base_device::DEVICE_CPU>;
    template struct apply_eigenvalues_op<double, base_device::DEVICE_CPU>;
    template struct precondition_op<std::complex<float>, base_device::DEVICE_CPU>;
    template struct precondition_op<std::complex<double>, base_device::DEVICE_CPU>;
    template struct precondition_op<double, base_device::DEVICE_CPU>;
    template struct normalize_op<std::complex<float>, base_device::DEVICE_CPU>;
    template struct normalize_op<std::complex<double>, base_device::DEVICE_CPU>;
    template struct normalize_op<double, base_device::DEVICE_CPU>;
    template struct refresh_hcc_scc_vcc_op<std::complex<float>, base_device::DEVICE_CPU>;
    template struct refresh_hcc_scc_vcc_op<std::complex<double>, base_device::DEVICE_CPU>;
    template struct refresh_hcc_scc_vcc_op<double, base_device::DEVICE_CPU>;
} // namespace hsolver
