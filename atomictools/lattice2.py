import chainer.functions as F

def cartesian_prod(*args):
    n = len(args)
    shapes = [tuple(len(x) if i == j else 1 for i in range(n)) for j, x in enumerate(args)]
    x = tuple(a.reshape(shape) for a, shape in zip(args, shapes))
    broad = F.broadcast(*tuple(a.reshape(shape) for a, shape in zip(args, shapes)))
    cat = F.concat([F.expand_dims(b, -1) for b in broad], axis=-1)
    return cat.reshape((-1, n))


def compute_shifts(cell, pbc, cutoff):
    xp = cell.xp
    reciprocal_cell = F.batch_inv(cell)
    inv_distances = F.max(F.sqrt(F.sum(reciprocal_cell ** 2, axis=1)), axis=0)
    num_repeats = F.ceil(cutoff * inv_distances)
    num_repeats = F.where(pbc, num_repeats, xp.zeros_like(num_repeats.data))
    num_repeats = F.max(num_repeats, axis=0)
    r1 = xp.arange(1, num_repeats.data[0] + 1)
    r2 = xp.arange(1, num_repeats.data[1] + 1)
    r3 = xp.arange(1, num_repeats.data[2] + 1)
    o = xp.zeros(1, dtype=r1.dtype)
    return F.vstack([
        xp.array([[0.0, 0.0, 0.0]]),
        cartesian_prod(r1, r2, r3),
        cartesian_prod(r1, r2, o),
        cartesian_prod(r1, r2, -r3),
        cartesian_prod(r1, o, r3),
        cartesian_prod(r1, o, o),
        cartesian_prod(r1, o, -r3),
        cartesian_prod(r1, -r2, r3),
        cartesian_prod(r1, -r2, o),
        cartesian_prod(r1, -r2, -r3),
        cartesian_prod(o, r2, r3),
        cartesian_prod(o, r2, o),
        cartesian_prod(o, r2, -r3),
        cartesian_prod(o, o, r3),
    ]).data



def neighbor_pairs_chainer(padding_mask, coordinates, cell, pbc, shifts, cutoff):
    xp = cell.xp
    def combination(n_atoms):
        all_atoms = xp.arange(n_atoms)
        i, j = cartesian_prod(all_atoms, all_atoms).data.T
        not_same = i != j
        return i[not_same], j[not_same]
    n1_center, n2_center = combination(padding_mask.shape[1])  # n_comb
    adapt_pbc = xp.prod(xp.logical_and(pbc[:, None, :], shifts[None, :, :] == 0.0), axis=2).astype(xp.bool)  # n_batch x n_shift
    r1_center = coordinates[:, n1_center, :]  # n_batch x n_comb x 3
    r2_center = coordinates[:, n2_center, :]  # n_batch x n_comb x 3
    r1 = r1_center[:, :, None, :]
    r2 = r2_center[:, :, None, :] + (shifts @ cell)[:, None, :, :]
    r1, r2 = F.broadcast(r1, r2)  # n_batch x n_comb x n_shift x 3
    r12 = F.sqrt(F.sum((r1 - r2) ** 2, axis=3))  # n_batch x n_comb x n_shift
    n1 = F.broadcast_to(n1_center[None, :, None], r12.shape).data  # n_batch x n_comb x n_shift
    n2 = F.broadcast_to(n2_center[None, :, None], r12.shape).data  # n_batch x n_comb x n_shift
    adapt_cutoff = r12.data < cutoff  # n_batch x n_comb x n_shift
    adapt_mask = xp.logical_and(padding_mask[:, n1_center.data], padding_mask[:, n2_center.data])  # n_batch x n_comb
    adapt_cutoff, adapt_pbc, adapt_mask = F.broadcast(adapt_cutoff, adapt_pbc[:, None, :], adapt_mask[:, :, None])
    adapt = xp.logical_and(adapt_cutoff.data, adapt_mask.data, adapt_pbc.data)  # n_batch x n_comb x n_shift
    batch = F.broadcast_to(xp.arange(padding_mask.shape[0])[:, None, None], adapt.shape).data
    shifts = F.broadcast_to(shifts[None, None, :, :], (*adapt.shape, 3))
    batch = batch[adapt]
    n1 = n1[adapt]
    n2 = n2[adapt]
    return batch, n1, n2, r1[adapt, :], r2[adapt, :], shifts[adapt, :].data, r12[adapt]


def test_neighbor_pairs(padding_mask, r, cell, pbc, shifts, cutoff): 
    xp = np
    batch, n1, n2, r1, r2, shifts2, r12 = neighbor_pairs_chainer(padding_mask, r, cell, pbc, shifts, cutoff)
    print(xp.allclose(F.sqrt(F.sum((r1 - r2) ** 2, axis=1)).data, r12.data))
    print(xp.allclose(r.reshape((-1, 3))[n1 + batch * r.shape[1], :].data, r1.data))
    print(xp.allclose(r.reshape((-1, 3))[n2 + batch * r.shape[1], :].data + shifts2 @ cell.data, r2.data))
    
    
    
def distance_matrix(r):
    return F.sqrt(F.sum((r[:, :, None, :] - r[:, None, :, :]) ** 2, axis=-1))


def small_distance_matrices(r, cutoff):
    # r : n_batch x n_atoms x 3
    n_batch = r.shape[0]
    n_atoms = r.shape[1]
    d = distance_matrix(r).data  # n_batch x n_atoms x n_atoms
    sort_index = np.argsort(d, axis=2)
    sort_index_inv = np.argsort(sort_index, axis=2)
    sorted_distance = np.take_along_axis(d, sort_index, axis=2)

    in_cut = np.sort(sorted_distance, axis=2) < 0.6
    n_adaptable = np.sum(in_cut, axis=2)  # n_batch x n_atoms
    max_n = np.max(np.sum(in_cut, axis=2))
    i = sort_index[:, :, :max_n]
    broad_r = F.broadcast_to(r[:, None, :, :], (n_batch, n_atoms, n_atoms, 3))
    shrink_r = np.take_along_axis(broad_r, i[:, :, :, None], axis=2)
    dm = F.sqrt(F.sum((shrink_r[:, :, :, None, :] - shrink_r[:, :, None, :, :]) ** 2, axis=4))
    filter_seed, na = F.broadcast(np.arange(max_n)[None, None, :], n_adaptable[:, :, None])  # n_batch x n_atoms x n_small
    filt = filter_seed.data < na.data
    filt1, filt2 = F.broadcast(filt[:, :, :, None], filt[:, :, None, :])
    filt = np.logical_and(filt1.data, filt2.data)
    return dm, filt, sort_index[:, :, :max_n]
