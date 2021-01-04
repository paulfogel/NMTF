"""Non-negative matrix and tensor factorization basic functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20
# Initialize progressbar
from sklearn.utils.extmath import randomized_svd
import numpy as np
import math

from .nmtf_core import (
    NMFSolve,
    Leverage,
    NTFSolve,
    NTFStack,
    NTFSolveFast,
    StatusBoxTqdm,
    NMFDet,
    BuildClusters,
    GlobalSign,
)

import sys

if not hasattr(sys, "argv"):
    sys.argv = [""]

EPSILON = np.finfo(np.float32).eps
compatibility_flag = False


# TODO : - objectify
#        - Typing
#        - proper docstring format (var name: var type
#                                       var descritiion)


def nmf_init(input_matrix, m_mis, m_t0, m_w0, nc, tolerance, log_iter, my_status_box):
    """Initialize NMF components using NNSVD

    Parameters
    ----------
    input_matrix: Input matrix
    m_mis: Define missing values (0 = missing cell, 1 = real cell)
    m_t0: Initial left hand matrix (may be empty)
    m_w0: Initial right hand matrix (may be empty)
    nc: NMF rank

    Returns
    -------
    Mt: Left hand matrix
    Mw: Right hand matrix
    
    Reference
    ---------

    C. Boutsidis, E. Gallopoulos (2008) SVD based initialization: A head start for nonnegative matrix factorization
    Pattern Recognition Pattern Recognition Volume 41, Issue 4, April 2008, Pages 1350-1362


    """

    # TODO : add tolerance, log_iter, my_status_box to docstring

    n, p = input_matrix.shape
    m_mis = m_mis.astype(np.int)
    n_mmis = m_mis.shape[0]
    if n_mmis == 0:
        ID = np.where(np.isnan(input_matrix) == True)
        n_mmis = ID[0].size
        if n_mmis > 0:
            m_mis = np.isnan(input_matrix) == False
            m_mis = m_mis.astype(np.int)
            input_matrix[m_mis == 0] = 0

    nc = int(nc)
    mt = np.copy(m_t0)
    mw = np.copy(m_w0)
    if (mt.shape[0] == 0) or (mw.shape[0] == 0):
        if n_mmis == 0:
            t, d, w = randomized_svd(input_matrix, n_components=nc, n_iter="auto", random_state=None)
            mt = t
            mw = w.T
        else:
            mt, d, mw, m_mis, mmsr, mmsr2, add_message, err_message, cancel_pressed = r_svd_solve(
                input_matrix, m_mis, nc, tolerance, log_iter, 0, "", 200, 1, 1, 1, my_status_box
            )

    for k in range(0, nc):
        u1 = mt[:, k]
        u2 = -mt[:, k]
        u1[u1 < 0] = 0
        u2[u2 < 0] = 0
        v1 = mw[:, k]
        v2 = -mw[:, k]
        v1[v1 < 0] = 0
        v2[v2 < 0] = 0
        u1 = np.reshape(u1, (n, 1))
        v1 = np.reshape(v1, (1, p))
        u2 = np.reshape(u2, (n, 1))
        v2 = np.reshape(v2, (1, p))
        if np.linalg.norm(u1 @ v1) > np.linalg.norm(u2 @ v2):
            mt[:, k] = np.reshape(u1, n)
            mw[:, k] = np.reshape(v1, p)
        else:
            mt[:, k] = np.reshape(u2, n)
            mw[:, k] = np.reshape(v2, p)

    return mt, mw


def r_nmf_solve(
    input_matrix,
    m_mis,
    m_t0,
    m_w0,
    nc,
    tolerance,
    precision,
    log_iter,
    max_iterations,
    nmf_algo,
    nmf_fix_user_lhe,
    nmf_fix_user_rhe,
    nmf_max_interm,
    nmf_sparse_level,
    nmf_robust_resample_columns,
    nmf_robust_n_runs,
    nmf_calculate_leverage,
    nmf_use_robust_leverage,
    nmf_find_parts,
    nmf_find_centroids,
    nmf_kernel,
    nmf_reweigh_columns,
    nmf_priors,
    my_status_box,
):
    """Estimate left and right hand matrices (robust version)

    Parameters
    ----------
    input_matrix: Input matrix
    m_mis: Define missing values (0 = missing cell, 1 = real cell)
    m_t0: Initial left hand matrix
    m_w0: Initial right hand matrix
    nc: NMF rank
    tolerance: Convergence threshold
    precision: Replace 0-values in multiplication rules
    log_iter: Log results through iterations
    max_iterations: Max iterations
    nmf_algo: =1,3: Divergence; =2,4: Least squares;
    nmf_fix_user_lhe: = 1 => fixed left hand matrix columns
    nmf_fix_user_rhe: = 1 => fixed  right hand matrix columns
    nmf_max_interm: Max iterations for warmup multiplication rules
    nmf_sparse_level: Requested sparsity in terms of relative number of rows with 0 values in right hand matrix
    nmf_robust_resample_columns: Resample columns during bootstrap
    nmf_robust_n_runs: Number of bootstrap runs
    nmf_calculate_leverage: Calculate leverages
    nmf_use_robust_leverage: Calculate leverages based on robust max across factoring columns
    nmf_find_parts: Enforce convexity on left hand matrix
    nmf_find_centroids: Enforce convexity on right hand matrix
    nmf_kernel: Type of kernel used; 1: linear; 2: quadraitc; 3: radial
    nmf_reweigh_columns: Reweigh columns in 2nd step of parts-based NMF
    nmf_priors: Priors on right hand matrix

    Returns
    -------
         Mt: Left hand matrix
         Mw: Right hand matrix
         MtPct: Percent robust clustered rows
         MwPct: Percent robust clustered columns
         diff: Objective minimum achieved
         Mh: Convexity matrix
         flagNonconvex: Updated non-convexity flag on left hand matrix

    """

    # TODO : add my_status_box to docstring

    # Check parameter consistency (and correct if needed)
    add_message = []
    err_message = ""
    cancel_pressed = 0
    nc = int(nc)
    if nmf_fix_user_lhe * nmf_fix_user_rhe == 1:
        return m_t0, m_w0, np.array([]), np.array([]), 0, np.array([]), 0, add_message, err_message, cancel_pressed

    if (nc == 1) & (nmf_algo > 2):
        nmf_algo -= 2

    if nmf_algo <= 2:
        nmf_robust_n_runs = 0

    m_mis = m_mis.astype(np.int)
    n_m_mis = m_mis.shape[0]
    if n_m_mis == 0:
        nan_id = np.where(np.isnan(input_matrix))
        n_m_mis = nan_id[0].size
        if n_m_mis > 0:
            m_mis = ~np.isnan(input_matrix)
            m_mis = m_mis.astype(np.int)
            input_matrix[m_mis == 0] = 0
    else:
        input_matrix[m_mis == 0] = 0

    if nmf_robust_resample_columns > 0:
        input_matrix = np.copy(input_matrix).T
        if n_m_mis > 0:
            m_mis = np.copy(m_mis).T

        m_temp = np.copy(m_w0)
        m_w0 = np.copy(m_t0)
        m_t0 = m_temp
        nmf_fix_user_lh_etemp = nmf_fix_user_lhe
        nmf_fix_user_lhe = nmf_fix_user_rhe
        nmf_fix_user_rhe = nmf_fix_user_lh_etemp

    n, p = input_matrix.shape

    # TODO : specify expected error
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except:
        n_nmf_priors = 0

    nmf_robust_n_runs = int(nmf_robust_n_runs)
    mt_pct = np.nan
    mw_pct = np.nan
    flag_nonconvex = 0

    # Step 1: NMF
    status = "Step 1 - NMF Ncomp=" + str(nc) + ": "
    mt, mw, diffsup, mhsup, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = NMFSolve(
        input_matrix,
        m_mis,
        m_t0,
        m_w0,
        nc,
        tolerance,
        precision,
        log_iter,
        status,
        max_iterations,
        nmf_algo,
        nmf_fix_user_lhe,
        nmf_fix_user_rhe,
        nmf_max_interm,
        100,
        nmf_sparse_level,
        nmf_find_parts,
        nmf_find_centroids,
        nmf_kernel,
        nmf_reweigh_columns,
        nmf_priors,
        flag_nonconvex,
        add_message,
        my_status_box,
    )
    m_tsup = np.copy(mt)
    m_wsup = np.copy(mw)
    if (n_nmf_priors > 0) & (nmf_reweigh_columns > 0):
        #     Run again with fixed LHE & no priors
        status = "Step 1bis - NMF (fixed LHE) Ncomp=" + str(nc) + ": "
        mw = np.ones((p, nc)) / math.sqrt(p)
        mt, mw, diffsup, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = NMFSolve(
            input_matrix,
            m_mis,
            m_tsup,
            mw,
            nc,
            tolerance,
            precision,
            log_iter,
            status,
            max_iterations,
            nmf_algo,
            nc,
            0,
            nmf_max_interm,
            100,
            nmf_sparse_level,
            nmf_find_parts,
            nmf_find_centroids,
            nmf_kernel,
            0,
            nmf_priors,
            flag_nonconvex,
            add_message,
            my_status_box,
        )
        m_tsup = np.copy(mt)
        m_wsup = np.copy(mw)

    # Bootstrap to assess robust clustering
    if nmf_robust_n_runs > 1:
        #     Update Mwsup
        mw_pct = np.zeros((p, nc))
        mw_blk = np.zeros((p, nmf_robust_n_runs * nc))
        for i_bootstrap in range(0, nmf_robust_n_runs):
            boot = np.random.randint(n, size=n)
            status = (
                "Step 2 - "
                + "Boot "
                + str(i_bootstrap + 1)
                + "/"
                + str(nmf_robust_n_runs)
                + " NMF Ncomp="
                + str(nc)
                + ": "
            )
            if n_m_mis > 0:
                mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = NMFSolve(
                    input_matrix[boot, :],
                    m_mis[boot, :],
                    m_tsup[boot, :],
                    m_wsup,
                    nc,
                    1.0e-3,
                    precision,
                    log_iter,
                    status,
                    max_iterations,
                    nmf_algo,
                    nc,
                    0,
                    nmf_max_interm,
                    20,
                    nmf_sparse_level,
                    nmf_find_parts,
                    nmf_find_centroids,
                    nmf_kernel,
                    nmf_reweigh_columns,
                    nmf_priors,
                    flag_nonconvex,
                    add_message,
                    my_status_box,
                )
            else:
                mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = NMFSolve(
                    input_matrix[boot, :],
                    m_mis,
                    m_tsup[boot, :],
                    m_wsup,
                    nc,
                    1.0e-3,
                    precision,
                    log_iter,
                    status,
                    max_iterations,
                    nmf_algo,
                    nc,
                    0,
                    nmf_max_interm,
                    20,
                    nmf_sparse_level,
                    nmf_find_parts,
                    nmf_find_centroids,
                    nmf_kernel,
                    nmf_reweigh_columns,
                    nmf_priors,
                    flag_nonconvex,
                    add_message,
                    my_status_box,
                )

            for k in range(0, nc):
                mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] = mw[:, k]

            mwn = np.zeros((p, nc))
            for k in range(0, nc):
                if (nmf_algo == 2) | (nmf_algo == 4):
                    scale_mw = np.linalg.norm(mw_blk[:, k * nmf_robust_n_runs + i_bootstrap])
                else:
                    scale_mw = np.sum(mw_blk[:, k * nmf_robust_n_runs + i_bootstrap])

                if scale_mw > 0:
                    mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] = (
                        mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] / scale_mw
                    )

                mwn[:, k] = mw_blk[:, k * nmf_robust_n_runs + i_bootstrap]

            col_clust = np.zeros(p, dtype=int)
            if nmf_calculate_leverage > 0:
                mwn, add_message, err_message, cancel_pressed = Leverage(
                    mwn, nmf_use_robust_leverage, add_message, my_status_box
                )

            for j in range(0, p):
                col_clust[j] = np.argmax(np.array(mwn[j, :]))
                mw_pct[j, col_clust[j]] = mw_pct[j, col_clust[j]] + 1

        mw_pct = mw_pct / nmf_robust_n_runs

        #     Update Mtsup
        mt_pct = np.zeros((n, nc))
        for i_bootstrap in range(0, nmf_robust_n_runs):
            status = (
                "Step 3 - "
                + "Boot "
                + str(i_bootstrap + 1)
                + "/"
                + str(nmf_robust_n_runs)
                + " NMF Ncomp="
                + str(nc)
                + ": "
            )
            mw = np.zeros((p, nc))
            for k in range(0, nc):
                mw[:, k] = mw_blk[:, k * nmf_robust_n_runs + i_bootstrap]

            mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = NMFSolve(
                input_matrix,
                m_mis,
                m_tsup,
                mw,
                nc,
                1.0e-3,
                precision,
                log_iter,
                status,
                max_iterations,
                nmf_algo,
                0,
                nc,
                nmf_max_interm,
                20,
                nmf_sparse_level,
                nmf_find_parts,
                nmf_find_centroids,
                nmf_kernel,
                nmf_reweigh_columns,
                nmf_priors,
                flag_nonconvex,
                add_message,
                my_status_box,
            )
            row_clust = np.zeros(n, dtype=int)
            if nmf_calculate_leverage > 0:
                mtn, add_message, err_message, cancel_pressed = Leverage(
                    mt, nmf_use_robust_leverage, add_message, my_status_box
                )
            else:
                mtn = mt

            for i in range(0, n):
                row_clust[i] = np.argmax(mtn[i, :])
                mt_pct[i, row_clust[i]] = mt_pct[i, row_clust[i]] + 1

        mt_pct = mt_pct / nmf_robust_n_runs

    mt = m_tsup
    mw = m_wsup
    mh = mhsup
    diff = diffsup

    if nmf_robust_resample_columns > 0:
        m_temp = np.copy(mt)
        mt = np.copy(mw)
        mw = m_temp
        m_temp = np.copy(mt_pct)
        mt_pct = np.copy(mw_pct)
        mw_pct = m_temp

    return mt, mw, mt_pct, mw_pct, diff, mh, flag_nonconvex, add_message, err_message, cancel_pressed


def ntf_init(
    input_tensor,
    m_mis,
    mtx_mw,
    mb2,
    nc,
    tolerance,
    precision,
    log_iter,
    ntf_unimodal,
    ntf_left_components,
    ntf_right_components,
    ntf_block_components,
    n_blocks,
    my_status_box,
):
    """Initialize NTF components for HALS

    Parameters
    ----------
    input_tensor: Input tensor
    m_mis: Define missing values (0 = missing cell, 1 = real cell)
    mtx_mw: initialization of LHM in NMF(unstacked tensor), may be empty
    mb2: initialization of RHM of NMF(unstacked tensor), may be empty
    n_blocks: Number of NTF blocks
    nc: NTF rank
    tolerance: Convergence threshold
    precision: Replace 0-values in multiplication rules
    log_iter: Log results through iterations

    Returns
    -------
    Mt: Left hand matrix
    Mw: Right hand matrix
    Mb: Block hand matrix
    """

    # TODO : add missing aarguments to docstring
    add_message = []
    n, p = input_tensor.shape
    m_mis = m_mis.astype(np.int)
    n_m_mis = m_mis.shape[0]
    if n_m_mis == 0:
        nan_id = np.where(np.isnan(input_tensor))
        n_m_mis = nan_id[0].size
        if n_m_mis > 0:
            m_mis = ~np.isnan(input_tensor)
            m_mis = m_mis.astype(np.int)
            input_tensor[m_mis == 0] = 0

    nc = int(nc)
    n_blocks = int(n_blocks)
    status0 = "Step 1 - Quick NMF Ncomp=" + str(nc) + ": "
    m_stacked, m_mis_stacked = NTFStack(input_tensor, m_mis, n_blocks)
    nc2 = min(nc, n_blocks)  # factorization rank can't be > number of blocks
    if (mtx_mw.shape[0] == 0) or (mb2.shape[0] == 0):
        mtx_mw, mb2 = nmf_init(m_stacked, m_mis_stacked, np.array([]), np.array([]), nc2, tolerance, log_iter, my_status_box)
    # NOTE: NMFInit (NNSVD) should always be called to prevent initializing NMF with signed components.
    # Creates minor differences in AD clustering, correction non implemented in Galderma version
    if not compatibility_flag:
        mtx_mw, mb2 = nmf_init(m_stacked, m_mis_stacked, mtx_mw, mb2, nc2, tolerance, log_iter, my_status_box)
    else:
        print("In NTFInit, NMFInit was not called for the sake of compatibility with previous versions")

    # Quick NMF
    mtx_mw, mb2, diff, mh, dummy1, dummy2, add_message, err_message, cancel_pressed = NMFSolve(
        m_stacked,
        m_mis_stacked,
        mtx_mw,
        mb2,
        nc2,
        tolerance,
        precision,
        log_iter,
        status0,
        10,
        2,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        np.array([]),
        0,
        add_message,
        my_status_box,
    )

    # Factorize Left vectors and distribute multiple factors if nc2 < nc
    mt = np.zeros((n, nc))
    mw = np.zeros((int(p / n_blocks), nc))
    mb = np.zeros((n_blocks, nc))
    n_fact = int(np.ceil(nc / n_blocks))
    for k in range(0, nc2):
        my_status_box.update_status(delay=1, status="Start SVD...")
        u, d, v = randomized_svd(
            np.reshape(mtx_mw[:, k], (int(p / n_blocks), n)).T, n_components=n_fact, n_iter="auto", random_state=None
        )
        v = v.T
        my_status_box.update_status(delay=1, status="SVD completed")
        for i_fact in range(0, n_fact):
            ind = i_fact * n_blocks + k
            if ind < nc:
                u1 = u[:, i_fact]
                u2 = -u[:, i_fact]
                u1[u1 < 0] = 0
                u2[u2 < 0] = 0
                v1 = v[:, i_fact]
                v2 = -v[:, i_fact]
                v1[v1 < 0] = 0
                v2[v2 < 0] = 0
                u1 = np.reshape(u1, (n, 1))
                v1 = np.reshape(v1, (1, int(p / n_blocks)))
                u2 = np.reshape(u2, (n, 1))
                v2 = np.reshape(v2, (1, int(p / n_blocks)))
                if np.linalg.norm(u1 @ v1) > np.linalg.norm(u2 @ v2):
                    mt[:, ind] = np.reshape(u1, n)
                    mw[:, ind] = d[i_fact] * np.reshape(v1, int(p / n_blocks))
                else:
                    mt[:, ind] = np.reshape(u2, n)
                    mw[:, ind] = d[i_fact] * np.reshape(v2, int(p / n_blocks))

                mb[:, ind] = mb2[:, k]

    for k in range(0, nc):
        if (ntf_unimodal > 0) & (ntf_left_components > 0):
            #                 Enforce unimodal distribution
            tmax = np.argmax(mt[:, k])
            for i in range(tmax + 1, n):
                mt[i, k] = min(mt[i - 1, k], mt[i, k])

            for i in range(tmax - 1, -1, -1):
                mt[i, k] = min(mt[i + 1, k], mt[i, k])

        if (ntf_unimodal > 0) & (ntf_right_components > 0):
            #                 Enforce unimodal distribution
            wmax = np.argmax(mw[:, k])
            for j in range(wmax + 1, int(p / n_blocks)):
                mw[j, k] = min(mw[j - 1, k], mw[j, k])

            for j in range(wmax - 1, -1, -1):
                mw[j, k] = min(mw[j + 1, k], mw[j, k])

        if (ntf_unimodal > 0) & (ntf_block_components > 0):
            #                 Enforce unimodal distribution
            bmax = np.argmax(mb[:, k])
            for i_block in range(bmax + 1, n_blocks):
                mb[i_block, k] = min(mb[i_block - 1, k], mb[i_block, k])

            for i_block in range(bmax - 1, -1, -1):
                mb[i_block, k] = min(mb[i_block + 1, k], mb[i_block, k])

    return [mt, mw, mb, add_message, err_message, cancel_pressed]


def rNTFSolve(
    M,
    Mmis,
    Mt0,
    Mw0,
    Mb0,
    nc,
    tolerance,
    precision,
    LogIter,
    MaxIterations,
    NMFFixUserLHE,
    NMFFixUserRHE,
    NMFFixUserBHE,
    NMFAlgo,
    NMFRobustNRuns,
    NMFCalculateLeverage,
    NMFUseRobustLeverage,
    NTFFastHALS,
    NTFNIterations,
    NMFSparseLevel,
    NTFUnimodal,
    NTFSmooth,
    NTFLeftComponents,
    NTFRightComponents,
    NTFBlockComponents,
    NBlocks,
    NTFNConv,
    NMFPriors,
    myStatusBox,
):
    """Estimate NTF matrices (robust version)

     Input:
         m: Input matrix
         m_mis: Define missing values (0 = missing cell, 1 = real cell)
         Mt0: Initial left hand matrix
         Mw0: Initial right hand matrix
         Mb0: Initial block hand matrix
         nc: NTF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         log_iter: Log results through iterations
         max_iterations: Max iterations
         NMFFixUserLHE: fix left hand matrix columns: = 1, else = 0
         NMFFixUserRHE: fix  right hand matrix columns: = 1, else = 0
         NMFFixUserBHE: fix  block hand matrix columns: = 1, else = 0
         NMFAlgo: =5: Non-robust version, =6: Robust version
         NMFRobustNRuns: Number of bootstrap runs
         NMFCalculateLeverage: Calculate leverages
         NMFUseRobustLeverage: Calculate leverages based on robust max across factoring columns
         NTFFastHALS: Use Fast HALS (does not accept handle missing values and convolution)
         NTFNIterations: Warmup iterations for fast HALS
         NMFSparseLevel : sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
         NTFUnimodal: Apply Unimodal constraint on factoring vectors
         NTFSmooth: Apply Smooth constraint on factoring vectors
         NTFLeftComponents: Apply Unimodal/Smooth constraint on left hand matrix
         NTFRightComponents: Apply Unimodal/Smooth constraint on right hand matrix
         NTFBlockComponents: Apply Unimodal/Smooth constraint on block hand matrix
         NBlocks: Number of NTF blocks
         NTFNConv: Half-Size of the convolution window on 3rd-dimension of the tensor
         NMFPriors: Elements in Mw that should be updated (others remain 0)

         
     Output:
         Mt_conv: Convolutional Left hand matrix
         Mt: Left hand matrix
         Mw: Right hand matrix
         Mb: Block hand matrix
         MtPct: Percent robust clustered rows
         MwPct: Percent robust clustered columns
         diff : Objective minimum achieved
     """
    AddMessage = []
    ErrMessage = ""
    cancel_pressed = 0
    n, p0 = M.shape
    nc = int(nc)
    NBlocks = int(NBlocks)
    p = int(p0 / NBlocks)
    NTFNConv = int(NTFNConv)
    if NMFFixUserLHE * NMFFixUserRHE * NMFFixUserBHE == 1:
        return (
            np.zeros((n, nc * (2 * NTFNConv + 1))),
            Mt0,
            Mw0,
            Mb0,
            np.zeros((n, p0)),
            np.ones((n, nc)),
            np.ones((p, nc)),
            AddMessage,
            ErrMessage,
            cancel_pressed,
        )

    Mmis = Mmis.astype(np.int)
    n_Mmis = Mmis.shape[0]
    if n_Mmis == 0:
        ID = np.where(np.isnan(M) == True)
        n_Mmis = ID[0].size
        if n_Mmis > 0:
            Mmis = np.isnan(M) == False
            Mmis = Mmis.astype(np.int)
            M[Mmis == 0] = 0
    else:
        M[Mmis == 0] = 0

    NTFNIterations = int(NTFNIterations)
    NMFRobustNRuns = int(NMFRobustNRuns)
    Mt = np.copy(Mt0)
    Mw = np.copy(Mw0)
    Mb = np.copy(Mb0)
    Mt_conv = np.array([])

    # Check parameter consistency (and correct if needed)
    if (nc == 1) | (NMFAlgo == 5):
        NMFRobustNRuns = 0

    if NMFRobustNRuns == 0:
        MtPct = np.nan
        MwPct = np.nan

    if (n_Mmis > 0 or NTFNConv > 0 or NMFSparseLevel != 0) and NTFFastHALS > 0:
        NTFFastHALS = 0
        reverse2HALS = 1
    else:
        reverse2HALS = 0

    # Step 1: NTF
    Status0 = "Step 1 - NTF Ncomp=" + str(nc) + ": "
    if NTFFastHALS > 0:
        if NTFNIterations > 0:
            Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
                M,
                Mmis,
                Mt,
                Mw,
                Mb,
                nc,
                tolerance,
                LogIter,
                Status0,
                NTFNIterations,
                NMFFixUserLHE,
                NMFFixUserRHE,
                NMFFixUserBHE,
                0,
                NTFUnimodal,
                NTFSmooth,
                NTFLeftComponents,
                NTFRightComponents,
                NTFBlockComponents,
                NBlocks,
                NTFNConv,
                NMFPriors,
                myStatusBox,
            )

        Mt, Mw, Mb, diff, cancel_pressed = NTFSolveFast(
            M,
            Mmis,
            Mt,
            Mw,
            Mb,
            nc,
            tolerance,
            precision,
            LogIter,
            Status0,
            MaxIterations,
            NMFFixUserLHE,
            NMFFixUserRHE,
            NMFFixUserBHE,
            NTFUnimodal,
            NTFSmooth,
            NTFLeftComponents,
            NTFRightComponents,
            NTFBlockComponents,
            NBlocks,
            myStatusBox,
        )
    else:
        Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
            M,
            Mmis,
            Mt,
            Mw,
            Mb,
            nc,
            tolerance,
            LogIter,
            Status0,
            MaxIterations,
            NMFFixUserLHE,
            NMFFixUserRHE,
            NMFFixUserBHE,
            NMFSparseLevel,
            NTFUnimodal,
            NTFSmooth,
            NTFLeftComponents,
            NTFRightComponents,
            NTFBlockComponents,
            NBlocks,
            NTFNConv,
            NMFPriors,
            myStatusBox,
        )

    Mtsup = np.copy(Mt)
    Mwsup = np.copy(Mw)
    Mbsup = np.copy(Mb)
    diff_sup = diff
    # Bootstrap to assess robust clustering
    if NMFRobustNRuns > 1:
        #     Update Mwsup
        MwPct = np.zeros((p, nc))
        MwBlk = np.zeros((p, NMFRobustNRuns * nc))
        for iBootstrap in range(0, NMFRobustNRuns):
            Boot = np.random.randint(n, size=n)
            Status0 = (
                "Step 2 - " + "Boot " + str(iBootstrap + 1) + "/" + str(NMFRobustNRuns) + " NTF Ncomp=" + str(nc) + ": "
            )
            if NTFFastHALS > 0:
                if n_Mmis > 0:
                    Mt, Mw, Mb, diff, cancel_pressed = NTFSolveFast(
                        M[Boot, :],
                        Mmis[Boot, :],
                        Mtsup[Boot, :],
                        Mwsup,
                        Mb,
                        nc,
                        1.0e-3,
                        precision,
                        LogIter,
                        Status0,
                        MaxIterations,
                        1,
                        0,
                        NMFFixUserBHE,
                        NTFUnimodal,
                        NTFSmooth,
                        NTFLeftComponents,
                        NTFRightComponents,
                        NTFBlockComponents,
                        NBlocks,
                        myStatusBox,
                    )
                else:
                    Mt, Mw, Mb, diff, cancel_pressed = NTFSolveFast(
                        M[Boot, :],
                        np.array([]),
                        Mtsup[Boot, :],
                        Mwsup,
                        Mb,
                        nc,
                        1.0e-3,
                        precision,
                        LogIter,
                        Status0,
                        MaxIterations,
                        1,
                        0,
                        NMFFixUserBHE,
                        NTFUnimodal,
                        NTFSmooth,
                        NTFLeftComponents,
                        NTFRightComponents,
                        NTFBlockComponents,
                        NBlocks,
                        myStatusBox,
                    )
            else:
                if n_Mmis > 0:
                    Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
                        M[Boot, :],
                        Mmis[Boot, :],
                        Mtsup[Boot, :],
                        Mwsup,
                        Mb,
                        nc,
                        1.0e-3,
                        LogIter,
                        Status0,
                        MaxIterations,
                        1,
                        0,
                        NMFFixUserBHE,
                        NMFSparseLevel,
                        NTFUnimodal,
                        NTFSmooth,
                        NTFLeftComponents,
                        NTFRightComponents,
                        NTFBlockComponents,
                        NBlocks,
                        NTFNConv,
                        NMFPriors,
                        myStatusBox,
                    )
                else:
                    Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
                        M[Boot, :],
                        np.array([]),
                        Mtsup[Boot, :],
                        Mwsup,
                        Mb,
                        nc,
                        1.0e-3,
                        LogIter,
                        Status0,
                        MaxIterations,
                        1,
                        0,
                        NMFFixUserBHE,
                        NMFSparseLevel,
                        NTFUnimodal,
                        NTFSmooth,
                        NTFLeftComponents,
                        NTFRightComponents,
                        NTFBlockComponents,
                        NBlocks,
                        NTFNConv,
                        NMFPriors,
                        myStatusBox,
                    )

            for k in range(0, nc):
                MwBlk[:, k * NMFRobustNRuns + iBootstrap] = Mw[:, k]

            Mwn = np.zeros((p, nc))
            for k in range(0, nc):
                ScaleMw = np.linalg.norm(MwBlk[:, k * NMFRobustNRuns + iBootstrap])
                if ScaleMw > 0:
                    MwBlk[:, k * NMFRobustNRuns + iBootstrap] = MwBlk[:, k * NMFRobustNRuns + iBootstrap] / ScaleMw

                Mwn[:, k] = MwBlk[:, k * NMFRobustNRuns + iBootstrap]

            ColClust = np.zeros(p, dtype=int)
            if NMFCalculateLeverage > 0:
                Mwn, AddMessage, ErrMessage, cancel_pressed = Leverage(
                    Mwn, NMFUseRobustLeverage, AddMessage, myStatusBox
                )

            for j in range(0, p):
                ColClust[j] = np.argmax(np.array(Mwn[j, :]))
                MwPct[j, ColClust[j]] = MwPct[j, ColClust[j]] + 1

        MwPct = MwPct / NMFRobustNRuns

        #     Update Mtsup
        MtPct = np.zeros((n, nc))
        for iBootstrap in range(0, NMFRobustNRuns):
            Status0 = (
                "Step 3 - " + "Boot " + str(iBootstrap + 1) + "/" + str(NMFRobustNRuns) + " NTF Ncomp=" + str(nc) + ": "
            )
            Mw = np.zeros((p, nc))
            for k in range(0, nc):
                Mw[:, k] = MwBlk[:, k * NMFRobustNRuns + iBootstrap]

            if NTFFastHALS > 0:
                Mt, Mw, Mb, diff, cancel_pressed = NTFSolveFast(
                    M,
                    Mmis,
                    Mtsup,
                    Mw,
                    Mb,
                    nc,
                    1.0e-3,
                    precision,
                    LogIter,
                    Status0,
                    MaxIterations,
                    0,
                    1,
                    NMFFixUserBHE,
                    NTFUnimodal,
                    NTFSmooth,
                    NTFLeftComponents,
                    NTFRightComponents,
                    NTFBlockComponents,
                    NBlocks,
                    myStatusBox,
                )
            else:
                Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
                    M,
                    Mmis,
                    Mtsup,
                    Mw,
                    Mb,
                    nc,
                    1.0e-3,
                    LogIter,
                    Status0,
                    MaxIterations,
                    0,
                    1,
                    NMFFixUserBHE,
                    NMFSparseLevel,
                    NTFUnimodal,
                    NTFSmooth,
                    NTFLeftComponents,
                    NTFRightComponents,
                    NTFBlockComponents,
                    NBlocks,
                    NTFNConv,
                    NMFPriors,
                    myStatusBox,
                )

            RowClust = np.zeros(n, dtype=int)
            if NMFCalculateLeverage > 0:
                Mtn, AddMessage, ErrMessage, cancel_pressed = Leverage(
                    Mt, NMFUseRobustLeverage, AddMessage, myStatusBox
                )
            else:
                Mtn = Mt

            for i in range(0, n):
                RowClust[i] = np.argmax(Mtn[i, :])
                MtPct[i, RowClust[i]] = MtPct[i, RowClust[i]] + 1

        MtPct = MtPct / NMFRobustNRuns

    Mt = Mtsup
    Mw = Mwsup
    Mb = Mbsup
    diff = diff_sup
    if reverse2HALS > 0:
        AddMessage.insert(
            len(AddMessage),
            "Currently, Fast HALS cannot be applied with missing data or convolution window and was reversed to Simple HALS.",
        )

    return Mt_conv, Mt, Mw, Mb, MtPct, MwPct, diff, AddMessage, ErrMessage, cancel_pressed


def r_svd_solve(
    m,
    m_mis,
    nc,
    tolerance,
    log_iter,
    log_trials,
    status0,
    max_iterations,
    svd_algo,
    svd_coverage,
    svdn_trials,
    my_status_box,
):
    """Estimate SVD matrices (robust version)

     Input:
         m: Input matrix
         m_mis: Define missing values (0 = missing cell, 1 = real cell)
         nc: SVD rank
         tolerance: Convergence threshold
         log_iter: Log results through iterations
         log_trials: Log results through trials
         status0: Initial displayed status to be updated during iterations
         max_iterations: Max iterations
         svd_algo: =1: Non-robust version, =2: Robust version
         svd_coverage: Coverage non-outliers (robust version)
         svdn_trials: Number of trials (robust version)
     
     Output:
         mt: Left hand matrix
         mev: Scaling factors
         mw: Right hand matrix
         m_mis: Matrix of missing/flagged outliers
         mmsr: Vector of Residual SSQ
         mmsr2: Vector of Reidual variance

     Reference
     ---------

     L. Liu et al (2003) Robust singular value decomposition analysis of microarray data
     PNAS November 11, 2003 vol. 100 no. 23 13167–13172

    """

    add_message = []
    err_message = ""
    cancel_pressed = 0

    # m0 is the running matrix (to be factorized, initialized from m)
    m0 = np.copy(m)
    n, p = m0.shape
    m_mis = m_mis.astype(np.bool_)
    n_mmis = m_mis.shape[0]

    # !! ATTENTION !! replacing ALL == False by is False seems to change the algo behavior.
    if n_mmis > 0:
        m0[m_mis == False] = np.nan
    else:
        m_mis = np.isnan(m0) == False
        m_mis = m_mis.astype(np.bool_)
        n_mmis = m_mis.shape[0]

    trace0 = np.sum(m0[m_mis] ** 2)
    nc = int(nc)
    svdn_trials = int(svdn_trials)
    nxp = n * p
    nxpcov = int(round(nxp * svd_coverage, 0))
    mmsr = np.zeros(nc)
    mmsr2 = np.zeros(nc)
    mev = np.zeros(nc)
    if svd_algo == 2:
        max_trial = svdn_trials
    else:
        max_trial = 1

    mw = np.zeros((p, nc))
    mt = np.zeros((n, nc))
    mdiff = np.zeros((n, p))
    w = np.zeros(p)
    t = np.zeros(n)
    w_trial = np.zeros(p)
    t_trial = np.zeros(n)
    mmis_trial = np.zeros((n, p), dtype=np.bool)
    # Outer-reference m becomes local reference m, which is the running matrix within ALS/LTS loop.
    m = np.zeros((n, p))
    wnorm = np.zeros((p, n))
    tnorm = np.zeros((n, p))
    denomw = np.zeros(n)
    denomt = np.zeros(p)
    step_iter = math.ceil(max_iterations / 100)
    pbar_step = 100 * step_iter / max_iterations
    if (n_mmis == 0) & (svd_algo == 1):
        fast_code = 1
    else:
        fast_code = 0

    if (fast_code == 0) and (svd_algo == 1):
        denomw[np.count_nonzero(m_mis, axis=1) < 2] = np.nan
        denomt[np.count_nonzero(m_mis, axis=0) < 2] = np.nan

    for k in range(0, nc):
        for i_trial in range(0, max_trial):
            my_status_box.init_bar(delay=1)
            # Copy values of m0 into m
            m[:, :] = m0
            status1 = status0 + "Ncomp " + str(k + 1) + " Trial " + str(i_trial + 1) + ": "
            if svd_algo == 2:
                #         Select a random subset
                m = np.reshape(m, (nxp, 1))
                m[np.argsort(np.random.rand(nxp))[nxpcov:nxp]] = np.nan
                m = np.reshape(m, (n, p))

            m_mis[:, :] = np.isnan(m) == False

            #         Initialize w
            for j in range(0, p):
                w[j] = np.median(m[m_mis[:, j], j])

            if np.where(w > 0)[0].size == 0:
                w[:] = 1

            w /= np.linalg.norm(w)
            # Replace missing values by 0's before regression
            m[m_mis == False] = 0

            #         initialize t (LTS  =stochastic)
            if fast_code == 0:
                wnorm[:, :] = np.repeat(w[:, np.newaxis] ** 2, n, axis=1) * m_mis.T
                denomw[:] = np.sum(wnorm, axis=0)
                # Request at least 2 non-missing values to perform row regression
                if svd_algo == 2:
                    denomw[np.count_nonzero(m_mis, axis=1) < 2] = np.nan

                t[:] = m @ w / denomw
            else:
                t[:] = m @ w / np.linalg.norm(w) ** 2

            t[np.isnan(t) == True] = np.median(t[np.isnan(t) == False])

            if svd_algo == 2:
                mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                # Restore missing values instead of 0's
                m[m_mis == False] = m0[m_mis == False]
                m = np.reshape(m, (nxp, 1))
                m[np.argsort(np.reshape(mdiff, nxp))[nxpcov:nxp]] = np.nan
                m = np.reshape(m, (n, p))
                m_mis[:, :] = np.isnan(m) == False
                # Replace missing values by 0's before regression
                m[m_mis == False] = 0

            i_iter = 0
            cont = 1
            while (cont > 0) & (i_iter < max_iterations):
                #                 build w
                if fast_code == 0:
                    tnorm[:, :] = np.repeat(t[:, np.newaxis] ** 2, p, axis=1) * m_mis
                    denomt[:] = np.sum(tnorm, axis=0)
                    # Request at least 2 non-missing values to perform column regression
                    if svd_algo == 2:
                        denomt[np.count_nonzero(m_mis, axis=0) < 2] = np.nan

                    w[:] = m.T @ t / denomt
                else:
                    w[:] = m.T @ t / np.linalg.norm(t) ** 2

                w[np.isnan(w) == True] = np.median(w[np.isnan(w) == False])
                #                 normalize w
                w /= np.linalg.norm(w)
                if svd_algo == 2:
                    mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                    # Restore missing values instead of 0's
                    m[m_mis == False] = m0[m_mis == False]
                    m = np.reshape(m, (nxp, 1))
                    # Outliers resume to missing values
                    m[np.argsort(np.reshape(mdiff, nxp))[nxpcov:nxp]] = np.nan
                    m = np.reshape(m, (n, p))
                    m_mis[:, :] = np.isnan(m) == False
                    # Replace missing values by 0's before regression
                    m[m_mis == False] = 0

                #                 build t
                if fast_code == 0:
                    wnorm[:, :] = np.repeat(w[:, np.newaxis] ** 2, n, axis=1) * m_mis.T
                    denomw[:] = np.sum(wnorm, axis=0)
                    # Request at least 2 non-missing values to perform row regression
                    if svd_algo == 2:
                        denomw[np.count_nonzero(m_mis, axis=1) < 2] = np.nan

                    t[:] = m @ w / denomw
                else:
                    t[:] = m @ w / np.linalg.norm(w) ** 2

                t[np.isnan(t) == True] = np.median(t[np.isnan(t) == False])
                #                 note: only w is normalized within loop, t is normalized after convergence
                if svd_algo == 2:
                    mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                    # Restore missing values instead of 0's
                    m[m_mis == False] = m0[m_mis == False]
                    m = np.reshape(m, (nxp, 1))
                    # Outliers resume to missing values
                    m[np.argsort(np.reshape(mdiff, nxp))[nxpcov:nxp]] = np.nan
                    m = np.reshape(m, (n, p))
                    m_mis[:, :] = np.isnan(m) == False
                    # Replace missing values by 0's before regression
                    m[m_mis == False] = 0

                if i_iter % step_iter == 0:
                    if svd_algo == 1:
                        mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))

                    status = status1 + "Iteration: %s" % int(i_iter)
                    my_status_box.update_status(delay=1, status=status)
                    my_status_box.update_bar(delay=1, step=pbar_step)
                    if my_status_box.cancel_pressed:
                        cancel_pressed = 1
                        return [mt, mev, mw, m_mis, mmsr, mmsr2, add_message, err_message, cancel_pressed]

                    diff = np.linalg.norm(mdiff[m_mis]) ** 2 / np.where(m_mis)[0].size
                    if log_iter == 1:
                        if svd_algo == 2:
                            my_status_box.myPrint(
                                "Ncomp: "
                                + str(k)
                                + " Trial: "
                                + str(i_trial)
                                + " Iter: "
                                + str(i_iter)
                                + " MSR: "
                                + str(diff)
                            )
                        else:
                            my_status_box.myPrint("Ncomp: " + str(k) + " Iter: " + str(i_iter) + " MSR: " + str(diff))
                    # TODO : diff0 might not exist yet
                    if i_iter > 0 and abs(diff - diff0) / diff0 < tolerance:
                        cont = 0

                    diff0 = diff

                i_iter += 1

            # save trial
            # TODO : diff might not exist yet
            if i_trial == 0 or diff < diff_trial:
                best_trial = i_trial
                diff_trial = diff
                t_trial[:] = t
                w_trial[:] = w
                mmis_trial[:, :] = m_mis

            if log_trials == 1:
                my_status_box.myPrint("Ncomp: " + str(k) + " Trial: " + str(i_trial) + " MSR: " + str(diff))

        if log_trials:
            # TODO : best_trial might not exist yet
            my_status_box.myPrint("Ncomp: " + str(k) + " Best trial: " + str(best_trial) + " MSR: " + str(diff_trial))

        t[:] = t_trial
        w[:] = w_trial
        mw[:, k] = w
        #         compute eigen value
        if svd_algo == 2:
            #             Robust regression of m on tw`
            mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
            r_mdiff = np.argsort(np.reshape(mdiff, nxp))
            t /= np.linalg.norm(t)  # Normalize t
            mt[:, k] = t
            m_mis = np.reshape(m_mis, nxp)
            m_mis[r_mdiff[nxpcov:nxp]] = False
            ycells = np.reshape(m0, (nxp, 1))[m_mis]
            xcells = np.reshape(np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)), (nxp, 1))[m_mis]
            mev[k] = ycells.T @ xcells / np.linalg.norm(xcells) ** 2
            m_mis = np.reshape(m_mis, (n, p))
        else:
            mev[k] = np.linalg.norm(t)
            mt[:, k] = t / mev[k]  # normalize t

        if k == 0:
            mmsr[k] = mev[k] ** 2
        else:
            mmsr[k] = mmsr[k - 1] + mev[k] ** 2
            mmsr2[k] = mmsr[k] - mev[0] ** 2

        # m0 is deflated before calculating next component
        m0 = m0 - mev[k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k].T, (1, p))

    trace02 = trace0 - mev[0] ** 2
    mmsr = 1 - mmsr / trace0
    mmsr[mmsr > 1] = 1
    mmsr[mmsr < 0] = 0
    mmsr2 = 1 - mmsr2 / trace02
    mmsr2[mmsr2 > 1] = 1
    mmsr2[mmsr2 < 0] = 0
    if nc > 1:
        r_mev = np.argsort(-mev)
        mev = mev[r_mev]
        mw0 = mw
        mt0 = mt
        for k in range(0, nc):
            mw[:, k] = mw0[:, r_mev[k]]
            mt[:, k] = mt0[:, r_mev[k]]

    m_mis[:, :] = True
    m_mis[mmis_trial == False] = False
    # m_mis.astype(dtype=int)

    return [mt, mev, mw, m_mis, mmsr, mmsr2, add_message, err_message, cancel_pressed]


def non_negative_factorization(
    x,
    w=None,
    h=None,
    n_components=None,
    update_w=True,
    update_h=True,
    beta_loss="frobenius",
    use_hals=False,
    n_bootstrap=0,
    tol=1e-6,
    max_iter=150,
    max_iter_mult=20,
    regularization=None,
    sparsity=0,
    leverage="standard",
    convex=None,
    kernel="linear",
    skewness=False,
    null_priors=False,
    random_state=None,
    verbose=0,
):
    """Compute Non-negative Matrix Factorization (NMF)

    Find two non-negative matrices (w, h) such as x = w @ h.T + Error.
    This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is minimized with an alternating minimization of w
    and h.

    Parameters
    ----------

    x : array-like, shape (n_samples, n_features)
        Constant matrix.

    w : array-like, shape (n_samples, n_components)
        prior w
        If n_update_W == 0 , it is used as a constant, to solve for h only.

    h : array-like, shape (n_features, n_components)
        prior h
        If n_update_H = 0 , it is used as a constant, to solve for w only.

    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    update_w : boolean, default: True
        Update or keep w fixed

    update_h : boolean, default: True
        Update or keep h fixed

    beta_loss : string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler'}.
        Beta divergence to be minimized, measuring the distance between x
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss == 'kullback-leibler', the input
        matrix x cannot contain zeros.

    use_hals : boolean
        True -> HALS algorithm (note that convex and kullback-leibler loss opions are not supported)
        False-> Projected gradiant
    
    n_bootstrap : integer, default: 0
        Number of bootstrap runs.

    tol : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations.

    max_iter_mult : integer, default: 20
        Maximum number of iterations in multiplicative warm-up to projected gradient (beta_loss = 'frobenius' only).

    regularization :  None | 'components' | 'transformation'
        Select whether the regularization affects the components (h), the
        transformation (w) or none of them.

    sparsity : float, default: 0
        Sparsity target with 0 <= sparsity <= 1 representing either:
        - the % rows in w or h set to 0 (when use_hals = False)
        - the mean % rows per column in w or h set to 0 (when use_hals = True)

    leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of w and h rows on each component.

    convex :  None | 'components' | 'transformation', default None
        Apply convex constraint on w or h.

    kernel :  'linear', 'quadratic', 'radial', default 'linear'
        Can be set if convex = 'transformation'.

    null_priors : boolean, default False
        Cells of h with prior cells = 0 will not be updated.
        Can be set only if prior h has been defined.

    skewness : boolean, default False
        When solving mixture problems, columns of x at the extremities of the convex hull will be given largest weights.
        The column weight is a function of the skewness and its sign.
        The expected sign of the skewness is based on the skewness of w components, as returned by the first pass
        of a 2-steps convex NMF. Thus, during the first pass, skewness must be set to False.
        Can be set only if convex = 'transformation' and prior w and h have been defined.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------

    Estimator (dictionary) with following entries

    w : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    h : array-like, shape (n_features, n_components)
        Solution to the non-negative least squares problem.

    volume : scalar, volume occupied by w and h

    wb : array-like, shape (n_samples, n_components)
        Percent consistently clustered rows for each component.
        only if n_bootstrap > 0.

    hb : array-like, shape (n_features, n_components)
        Percent consistently clustered columns for each component.
        only if n_bootstrap > 0.

    b : array-like, shape (n_observations, n_components) or (n_features, n_components)
        only if active convex variant, h = B.T @ x or w = x @ B
    
    diff : Objective minimum achieved

        """

    if use_hals:
        # convex and kullback-leibler loss options are not supported
        beta_loss = "frobenius"
        convex = None

    n, p = x.shape
    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    if beta_loss == "frobenius":
        nmf_algo = 2
    else:
        nmf_algo = 1

    log_iter = verbose
    my_status_box = StatusBoxTqdm(verbose=log_iter)
    tolerance = tol
    precision = EPSILON

    if (w is None) & (h is None):
        mt, mw = nmf_init(x, np.array([]), np.array([]), np.array([]), nc, tolerance, log_iter, my_status_box)
        init = "nndsvd"
    else:
        if h is None:
            mw = np.ones((p, nc))
            init = "custom_W"
        elif w is None:
            mt = np.ones((n, nc))
            init = "custom_H"
        else:
            init = "custom"

        for k in range(0, nc):
            if nmf_algo == 2:
                # TODO : what if using custom ? mt/mw will not exist yet
                mt[:, k] = mt[:, k] / np.linalg.norm(mt[:, k])
                mw[:, k] = mw[:, k] / np.linalg.norm(mw[:, k])
            else:
                mt[:, k] = mt[:, k] / np.sum(mt[:, k])
                mw[:, k] = mw[:, k] / np.sum(mw[:, k])

    if n_bootstrap is None:
        n_bootstrap = 0

    if n_bootstrap > 1:
        nmf_algo += 2

    if update_w is True:
        update_w = 0
    else:
        update_w = 1

    if update_h is True:
        update_h = 0
    else:
        update_h = 1

    if regularization is None:
        regularization = 0
    else:
        if regularization == "components":
            regularization = sparsity
        elif regularization == "transformation":
            regularization = -sparsity
        else:
            regularization = 0

    nmf_robust_resample_columns = 0

    if leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif leverage == "robust":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 1
    else:
        nmf_calculate_leverage = 0
        nmf_use_robust_leverage = 0

    if convex is None:
        nmf_find_parts = 0
        nmf_find_centroids = 0
        nmf_kernel = 1
    elif convex == "transformation":
        nmf_find_parts = 1
        nmf_find_centroids = 0
        nmf_kernel = 1
    elif convex == "components":
        nmf_find_parts = 0
        nmf_find_centroids = 1
        if kernel == "linear":
            nmf_kernel = 1
        elif kernel == "quadratic":
            nmf_kernel = 2
        elif kernel == "radial":
            nmf_kernel = 3
        else:
            nmf_kernel = 1
    else:
        raise ValueError(f"Incorrect value for convex : {convex}")

    if (null_priors is True) & ((init == "custom") | (init == "custom_H")):
        nmf_priors = h
    else:
        nmf_priors = np.array([])

    if convex is None:
        nmf_reweigh_columns = 0
    else:
        if (convex == "transformation") & (init == "custom"):
            if skewness is True:
                nmf_reweigh_columns = 1
            else:
                nmf_reweigh_columns = 0

        else:
            nmf_reweigh_columns = 0

    if random_state is not None:
        random_seed = random_state
        np.random.seed(random_seed)

    if use_hals:
        if nmf_algo <= 2:
            ntf_algo = 5
        else:
            ntf_algo = 6

        mt_conv, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed = rNTFSolve(
            x,
            np.array([]),
            mt,
            mw,
            np.array([]),
            nc,
            tolerance,
            precision,
            log_iter,
            max_iter,
            update_w,
            update_h,
            1,
            ntf_algo,
            n_bootstrap,
            nmf_calculate_leverage,
            nmf_use_robust_leverage,
            0,
            0,
            regularization,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            np.array([]),
            my_status_box,
        )
        mev = np.ones(nc)
        if (update_w == 0) & (update_h == 0):
            # Scale
            for k in range(0, nc):
                scale_mt = np.linalg.norm(mt[:, k])
                scale_mw = np.linalg.norm(mw[:, k])
                mev[k] = scale_mt * scale_mw
                if mev[k] > 0:
                    mt[:, k] = mt[:, k] / scale_mt
                    mw[:, k] = mw[:, k] / scale_mw

    else:
        mt, mw, mt_pct, mw_pct, diff, mh, flag_nonconvex, add_message, err_message, cancel_pressed = r_nmf_solve(
            x,
            np.array([]),
            mt,
            mw,
            nc,
            tolerance,
            precision,
            log_iter,
            max_iter,
            nmf_algo,
            update_w,
            update_h,
            max_iter_mult,
            regularization,
            nmf_robust_resample_columns,
            n_bootstrap,
            nmf_calculate_leverage,
            nmf_use_robust_leverage,
            nmf_find_parts,
            nmf_find_centroids,
            nmf_kernel,
            nmf_reweigh_columns,
            nmf_priors,
            my_status_box,
        )

        mev = np.ones(nc)
        if (nmf_find_parts == 0) & (nmf_find_centroids == 0) & (update_w == 0) & (update_h == 0):
            # Scale
            for k in range(0, nc):
                if (nmf_algo == 2) | (nmf_algo == 4):
                    scale_mt = np.linalg.norm(mt[:, k])
                    scale_mw = np.linalg.norm(mw[:, k])
                else:
                    scale_mt = np.sum(mt[:, k])
                    scale_mw = np.sum(mw[:, k])

                mev[k] = scale_mt * scale_mw
                if mev[k] > 0:
                    mt[:, k] = mt[:, k] / scale_mt
                    mw[:, k] = mw[:, k] / scale_mw

    volume = NMFDet(mt, mw, 1)

    for message in add_message:
        print(message)

    my_status_box.close()

    # Order by decreasing scale
    r_mev = np.argsort(-mev)
    mev = mev[r_mev]
    mt = mt[:, r_mev]
    mw = mw[:, r_mev]
    if isinstance(mt_pct, np.ndarray):
        mt_pct = mt_pct[:, r_mev]
        mw_pct = mw_pct[:, r_mev]

    if (nmf_find_parts == 0) & (nmf_find_centroids == 0):
        # Scale by max com p
        for k in range(0, nc):
            max_col = np.max(mt[:, k])
            if max_col > 0:
                mt[:, k] /= max_col
                mw[:, k] *= mev[k] * max_col
                mev[k] = 1
            else:
                mev[k] = 0

    estimator = {}
    if n_bootstrap <= 1:
        if (nmf_find_parts == 0) & (nmf_find_centroids == 0):
            estimator.update([("w", mt), ("h", mw), ("volume", volume), ("diff", diff)])
        else:
            # TODO : mh can be undefined
            estimator.update([("w", mt), ("h", mw), ("volume", volume), ("b", mh), ("diff", diff)])

    else:
        # TODO : mh can be undefined
        if (nmf_find_parts == 0) & (nmf_find_centroids == 0):
            estimator.update([("w", mt), ("h", mw), ("volume", volume), ("wb", mt_pct), ("hb", mw_pct), ("diff", diff)])
        else:
            estimator.update(
                [("w", mt), ("h", mw), ("volume", volume), ("b", mh), ("wb", mt_pct), ("hb", mw_pct), ("diff", diff)]
            )

    return estimator


def nmf_predict(estimator, leverage="robust", blocks=None, cluster_by_stability=False, custom_order=False, verbose=0):
    """Derives ordered sample and feature indexes for future use in ordered heatmaps

    Parameters
    ----------

    estimator : tuplet as returned by non_negative_factorization

    leverage :  None | 'standard' | 'robust', default 'robust'
        Calculate leverage of w and h rows on each component.

    blocks : array-like, shape(n_blocks), default None
        Size of each block (if any) in ordered heatmap.

    cluster_by_stability : boolean, default False
         Use stability instead of leverage to assign samples/features to clusters

    custom_order :  boolean, default False
         if False samples/features with highest leverage or stability appear on top of each cluster
         if True within cluster ordering is modified to suggest a continuum  between adjacent clusters

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------

    Completed estimator with following entries:
    wl : array-like, shape (n_samples, n_components)
         Sample leverage on each component

    hl : array-like, shape (n_features, n_components)
         Feature leverage on each component

    ql : array-like, shape (n_blocks, n_components)
         Block leverage on each component (NTF only)

    wr : vector-like, shape (n_samples)
         Ranked sample indexes (by cluster and leverage or stability)
         Used to produce ordered heatmaps

    hr : vector-like, shape (n_features)
         Ranked feature indexes (by cluster and leverage or stability)
         Used to produce ordered heatmaps

    wn : vector-like, shape (n_components)
         Sample cluster bounds in ordered heatmap

    hn : vector-like, shape (n_components)
         Feature cluster bounds in ordered heatmap

    wc : vector-like, shape (n_samples)
         Sample assigned cluster

    hc : vector-like, shape (n_features)
         Feature assigned cluster

    qc : vector-like, shape (size(blocks))
         Block assigned cluster (NTF only)

    """

    mt = estimator["w"]
    mw = estimator["h"]
    if "q" in estimator:
        # x is a 3D tensor, in unfolded form of a 2D array
        # horizontal concatenation of blocks of equal size.
        mb = estimator["q"]
        nmf_algo = 5
        n_blocks = mb.shape[0]
        blk_size = mw.shape[0] * np.ones(n_blocks)
    else:
        mb = np.array([])
        nmf_algo = 0
        if blocks is None:
            n_blocks = 1
            blk_size = np.array([mw.shape[0]])
        else:
            n_blocks = blocks.shape[0]
            blk_size = blocks

    if "wb" in estimator:
        mt_pct = estimator["wb"]
    else:
        mt_pct = None

    if "hb" in estimator:
        mw_pct = estimator["hb"]
    else:
        mw_pct = None

    if leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif leverage == "robust":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 1
    else:
        nmf_calculate_leverage = 0
        nmf_use_robust_leverage = 0

    if cluster_by_stability is True:
        nmf_robust_cluster_by_stability = 1
    else:
        nmf_robust_cluster_by_stability = 0

    if custom_order is True:
        cell_plot_ordered_clusters = 1
    else:
        cell_plot_ordered_clusters = 0

    add_message = []
    my_status_box = StatusBoxTqdm(verbose=verbose)

    (
        mtn,
        mwn,
        mbn,
        r_ct,
        r_cw,
        n_ct,
        n_cw,
        row_clust,
        col_clust,
        block_clust,
        add_message,
        err_message,
        cancel_pressed,
    ) = BuildClusters(
        mt,
        mw,
        mb,
        mt_pct,
        mw_pct,
        n_blocks,
        blk_size,
        nmf_calculate_leverage,
        nmf_use_robust_leverage,
        nmf_algo,
        nmf_robust_cluster_by_stability,
        cell_plot_ordered_clusters,
        add_message,
        my_status_box,
    )
    for message in add_message:
        print(message)

    my_status_box.close()
    if "q" in estimator:
        estimator.update(
            [
                ("wl", mtn),
                ("hl", mwn),
                ("wr", r_ct),
                ("hr", r_cw),
                ("wn", n_ct),
                ("hn", n_cw),
                ("wc", row_clust),
                ("hc", col_clust),
                ("ql", mbn),
                ("qc", block_clust),
            ]
        )
    else:
        estimator.update(
            [
                ("wl", mtn),
                ("hl", mwn),
                ("wr", r_ct),
                ("hr", r_cw),
                ("wn", n_ct),
                ("hn", n_cw),
                ("wc", row_clust),
                ("hc", col_clust),
                ("ql", None),
                ("qc", None),
            ]
        )
    return estimator


def nmf_permutation_test_score(estimator, y, n_permutations=100, verbose=0):
    """Do a permutation test to assess association between ordered samples and some covariate

    Parameters
    ----------

    estimator : tuplet as returned by non_negative_factorization and nmf_predict

    y :  array-like, group to be predicted

    n_permutations :  integer, default: 100

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------

    Completed estimator with following entries:

    score : float
         The true score without permuting targets.

    pvalue : float
         The p-value, which approximates the probability that the score would be obtained by chance.

    CS : array-like, shape(n_components)
         The size of each cluster

    CP : array-like, shape(n_components)
         The pvalue of the most significant group within each cluster

    CG : array-like, shape(n_components)
         The index of the most significant group within each cluster

    CN : array-like, shape(n_components, n_groups)
         The size of each group within each cluster


    """
    mt = estimator["w"]
    r_ct = estimator["wr"]
    n_ct = estimator["wn"]
    row_groups = y
    uniques, index = np.unique([row for row in row_groups], return_index=True)
    list_groups = row_groups[index]
    nb_groups = list_groups.shape[0]
    ngroup = np.zeros(nb_groups)
    for group in range(0, nb_groups):
        ngroup[group] = np.where(row_groups == list_groups[group])[0].shape[0]

    nrun = n_permutations
    my_status_box = StatusBoxTqdm(verbose=verbose)
    cluster_size, pglob, prun, cluster_prob, cluster_group, cluster_ngroup, cancel_pressed = GlobalSign(
        nrun, nb_groups, mt, r_ct, n_ct, row_groups, list_groups, ngroup, my_status_box
    )

    estimator.update(
        [
            ("score", prun),
            ("pvalue", pglob),
            ("cs", cluster_size),
            ("cp", cluster_prob),
            ("cg", cluster_group),
            ("cn", cluster_ngroup),
        ]
    )
    return estimator


def non_negative_tensor_factorization(
    x,
    n_blocks,
    w=None,
    h=None,
    q=None,
    n_components=None,
    update_w=True,
    update_h=True,
    update_q=True,
    fast_hals=True,
    n_iter_hals=2,
    n_shift=0,
    regularization=None,
    sparsity=0,
    unimodal=False,
    smooth=False,
    apply_left=False,
    apply_right=False,
    apply_block=False,
    n_bootstrap=None,
    tol=1e-6,
    max_iter=150,
    leverage="standard",
    random_state=None,
    verbose=0,
):
    """Compute Non-negative Tensor Factorization (NTF)

    Find three non-negative matrices (w, h, F) such as x = w @@ h @@ F + Error (@@ = tensor product).
    This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is minimized with an alternating minimization of w
    and h.

    Parameters
    ----------

    x : array-like, shape (n_samples, n_features x n_blocks)
        Constant matrix.
        X is a tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.

    n_blocks : integer

    w : array-like, shape (n_samples, n_components)
        prior w

    h : array-like, shape (n_features, n_components)
        prior h

    q : array-like, shape (n_blocks, n_components)
        prior Q

    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    update_w : boolean, default: True
        Update or keep w fixed

    update_h : boolean, default: True
        Update or keep h fixed

    update_q : boolean, default: True
        Update or keep Q fixed

    fast_hals : boolean, default: True
        Use fast implementation of HALS

    n_iter_hals : integer, default: 2
        Number of HALS iterations prior to fast HALS
    
    n_shift : integer, default: 0
        max shifting in convolutional NTF

    regularization :  None | 'components' | 'transformation'
        Select whether the regularization affects the components (h), the
        transformation (w) or none of them.

    sparsity : float, default: 0
        Sparsity target with 0 <= sparsity <= 1 representing the mean % rows per column in w or h set to 0
  
    unimodal : Boolean, default: False

    smooth : Boolean, default: False

    apply_left : Boolean, default: False

    apply_right : Boolean, default: False

    apply_block : Boolean, default: False

    n_bootstrap : integer, default: 0
        Number of bootstrap runs.

    tol : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations.

    leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of w and h rows on each component.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------

        Estimator (dictionary) with following entries

        w : array-like, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.

        h : array-like, shape (n_features, n_components)
            Solution to the non-negative least squares problem.

        q : array-like, shape (n_blocks, n_components)
            Solution to the non-negative least squares problem.
               
        volume : scalar, volume occupied by w and h
        
        wb : array-like, shape (n_samples, n_components)
            Percent consistently clustered rows for each component.
            only if n_bootstrap > 0.

        hb : array-like, shape (n_features, n_components)
            Percent consistently clustered columns for each component.
            only if n_bootstrap > 0.

    Reference
    ---------

    A. Cichocki, P.h.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

    """

    n, p = x.shape
    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    log_iter = verbose
    my_status_box = StatusBoxTqdm(verbose=log_iter)
    tolerance = tol
    precision = EPSILON

    p_block = int(p / n_blocks)

    if regularization is None:
        nmf_sparse_level = 0
    else:
        if regularization == "components":
            nmf_sparse_level = sparsity
        elif regularization == "transformation":
            nmf_sparse_level = -sparsity
        else:
            nmf_sparse_level = 0
    if random_state is not None:
        np.random.seed(random_state)

    if (w is None) & (h is None) & (q is None):
        mt0, mw0, mb0, add_message, err_message, cancel_pressed = ntf_init(
            x,
            np.array([]),
            np.array([]),
            np.array([]),
            nc,
            tolerance,
            precision,
            log_iter,
            unimodal,
            apply_left,
            apply_right,
            apply_block,
            n_blocks,
            my_status_box,
        )
    else:
        if w is None:
            mt0 = np.ones((n, nc))
        else:
            mt0 = np.copy(w)

        if h is None:
            mw0 = np.ones((p_block, nc))
        else:
            mw0 = np.copy(h)

        if q is None:
            mb0 = np.ones((n_blocks, nc))
        else:
            mb0 = np.copy(q)

        mfit = np.zeros((n, p))
        for k in range(0, nc):
            for i_block in range(0, n_blocks):
                mfit[:, i_block * p_block : (i_block + 1) * p_block] += (
                    mb0[i_block, k] * np.reshape(mt0[:, k], (n, 1)) @ np.reshape(mw0[:, k], (1, p_block))
                )

        scale_ratio = (np.linalg.norm(mfit) / np.linalg.norm(x)) ** (1 / 3)
        for k in range(0, nc):
            mt0[:, k] /= scale_ratio
            mw0[:, k] /= scale_ratio
            mb0[:, k] /= scale_ratio

        mfit = np.zeros((n, p))
        for k in range(0, nc):
            for i_block in range(0, n_blocks):
                mfit[:, i_block * p_block : (i_block + 1) * p_block] += (
                    mb0[i_block, k] * np.reshape(mt0[:, k], (n, 1)) @ np.reshape(mw0[:, k], (1, p_block))
                )

    if n_bootstrap is None:
        n_bootstrap = 0

    if n_bootstrap <= 1:
        nmf_algo = 5
    else:
        nmf_algo = 6

    if leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif leverage == "robust":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 1
    else:
        nmf_calculate_leverage = 0
        nmf_use_robust_leverage = 0

    if random_state is not None:
        np.random.seed(random_state)

    if update_w:
        update_w = 0
    else:
        update_w = 1

    if update_h:
        update_h = 0
    else:
        update_h = 1

    if update_q:
        update_q = 0
    else:
        update_q = 1

    mt_conv, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed = rNTFSolve(
        x,
        np.array([]),
        mt0,
        mw0,
        mb0,
        nc,
        tolerance,
        precision,
        log_iter,
        max_iter,
        update_w,
        update_h,
        update_q,
        nmf_algo,
        n_bootstrap,
        nmf_calculate_leverage,
        nmf_use_robust_leverage,
        fast_hals,
        n_iter_hals,
        nmf_sparse_level,
        unimodal,
        smooth,
        apply_left,
        apply_right,
        apply_block,
        n_blocks,
        n_shift,
        np.array([]),
        my_status_box,
    )

    volume = NMFDet(mt, mw, 1)

    for message in add_message:
        print(message)

    my_status_box.close()

    estimator = {}
    if n_bootstrap <= 1:
        estimator.update([("w", mt), ("h", mw), ("q", mb), ("volume", volume), ("diff", diff)])
    else:
        estimator.update(
            [("w", mt), ("h", mw), ("q", mb), ("volume", volume), ("wb", mt_pct), ("hb", mw_pct), ("diff", diff)]
        )

    return estimator
