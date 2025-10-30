# Caching Unification Follow-Ups

- [x] Port remaining kernel modules to implement `CacheableKernel` directly rather than via legacy wrappers (completed: kernels now own their cache structs, `cacheable_resources/*` removed).
- [x] Eliminate shared helper glue (`mps_data_type_for_dtype`, KV write wrappers) so cache modules rely solely on `Dtype` conversions and local definitions.
- [ ] Expand the registry to cover KV cache and tensor preparation caches so that all cache metrics flow through a single path.
- [ ] Introduce eviction policies for oversized caches (consult `docs/IMPROVED_CACHING_INFRA.md` Phase 4).
- [ ] Document the new cache APIs in developer onboarding materials and update existing call-site comments.
