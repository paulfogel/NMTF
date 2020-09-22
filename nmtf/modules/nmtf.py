""" Classes accessing Non-negative matrix and tensor factorization functions

"""

# Author: Paul Fogel

# License: MIT
# Dec 28, '19

from .nmtf_base import (
    non_negative_factorization,
    nmf_predict,
    nmf_permutation_test_score,
    non_negative_tensor_factorization,
)
from typing import Union, List
import inspect
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\boldmath'


class NMF:
    """Initialize NMF model

    Parameters
    ----------
    n_components: int
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    n_update_w: int
        Estimate last n_update_w components from initial guesses.
        If n_update_w is not set : n_update_W = n_components.

    n_update_h: int
        Estimate last n_update_h components from initial guesses.
        If n_update_h is not set : n_update_H = n_components.

    beta_loss: str, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler'}.
        Beta divergence to be minimized, measuring the distance between x
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss == 'kullback-leibler', the input
        matrix x cannot contain zeros.
    
    use_hals: bool
        True -> HALS algorithm (note that convex & kullback-leibler loss options are not supported)
        False-> Projected gradiant

    tol: float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter: int, default: 200
        Maximum number of iterations.

    max_iter_mult: int, default: 20
        Maximum number of iterations in multiplicative warm-up to projected gradient (beta_loss = 'frobenius' only).

    leverage: str
        None | 'standard' | 'robust', default 'standard'
        Calculate leverage of W and H rows on each component.

    convex:  None | 'components' | 'transformation', default None
        Apply convex constraint on W or H.

    kernel: str
        'linear', 'quadratic', 'radial', default 'linear'
        Can be set if convex = 'transformation'.

    random_state: int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, default: 0
        The verbosity level (0/1).

    x: Union[np.ndarray, None]
        The x matrix to factorize, Can not be specified if w or h was spacified

    w_input: Union[np.ndarray, None]
        The cluster matrix from which to generate x. Can not be specified if x was specified. Need to specify also h

    h_input: Union[np.ndarray, None]
        The weight matrix from which to generate x. Can not be specified if x was specified. Need to specify also w

    **kwargs: tuple
        Any keyword argument to pass to fit_transform or predict

    Returns
    -------
    NMF model

    Example
    -------
    >>> from nmtf import NMF
    >>> import numpy
    >>> w = numpy.array([[1, 2],
    >>>                 [3, 4],
    >>>                 [5, 6],
    >>>                 [7, 8],
    >>>                 [9, 10],
    >>>                 [11, 12]])
    >>> h = numpy.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    >>>                 [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    >>> x = w.dot(h)
    >>> # Use either one of the following 2 lines
    >>> my_nmf_model = NMF(n_components=2, w_input=w, h_input=h)
    >>> # my_nmf_model = NMF(n_components=2, x=x)
    >>> estimator = my_nmf_model.predict(estimator)
    >>> # sampleGroup: the group each sample is associated with
    >>> # estimator = my_nmf_model.permutation_test_score(estimator, RowGroups, n_permutations=100)
    >>>
    >>> my_nmf_model.fit_transform()
    >>> my_nmf_model.plot()
    >>> my_nmf_model.fig_result.savefig("result_ignore.pdf")
    >>> my_nmf_model.fig_input.savefig("input_ignore.pdf")
    >>> my_nmf_model.fig_diff.savefig("diff_ignore.pdf")
    >>> my_nmf_model.fig_diff_l.savefig("diff_l_ignore.pdf")

    References
    ----------
        
        P. Fogel, D.M. Hawkins, C. Beecher, G. Luta, S. S. Young (2013). A Tale of Two Matrix Factorizations.
        The American Statistician, Vol. 67, Issue 4.

        C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
        IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1

    """

    def __init__(
        self,
        n_components: int = None,
        n_update_w: int = None,
        n_update_h: int = None,
        beta_loss: str = "frobenius",
        use_hals: bool = False,
        tol: float = 1e-6,
        max_iter=150,
        max_iter_mult=20,
        leverage="standard",
        convex=None,
        kernel="linear",
        random_state=None,
        verbose=0,
        x: Union[np.ndarray, None] = None,
        w_input: Union[np.ndarray, None] = None,
        h_input: Union[np.ndarray, None] = None,
        **kwargs
    ):

        if h_input is None and w_input is not None:
            raise ValueError("You specified w but not h. Need both!")
        if w_input is None and h_input is not None:
            raise ValueError("You specified h but not w. Need both!")
        if x is None and w_input is None:  # if w is None so is h, otherwise would have raised already
            raise ValueError("Need x or w and h!")
        if x is not None and w_input is not None:  # if w is not None neither is h, otherwise would have raised already
            raise ValueError("Can not specify both x and w/h!")
        if x is not None:
            self.x = x
            self.h_input = None
            self.w_input = None
        if w_input is not None and h_input is not None:
            self.w_input = w_input
            self.h_input = h_input
            try:
                self.x = w_input.dot(h_input)
            except ValueError:
                self.x = w_input.dot(h_input.T)

        self.n_components = n_components
        self.n_update_w = n_update_w
        self.n_update_h = n_update_h
        self.beta_loss = beta_loss
        self.use_hals = use_hals
        self.tol = tol
        self.max_iter = max_iter
        self.max_iter_mult = max_iter_mult
        self.leverage = leverage
        self.convex = convex
        self.kernel = kernel
        self.random_state = random_state
        self.verbose = verbose
        self.x_index_name = "Values"
        self.x_cols_name = "Features"

        args_for_fit = {}
        for arg in inspect.getfullargspec(NMF.fit_transform).args:
            if arg in kwargs:
                args_for_fit[arg] = kwargs[arg]

        self.estimator = self.fit_transform(**args_for_fit)

        args_for_predict = {}
        for arg in inspect.getfullargspec(NMF.predict).args:
            if arg in kwargs:
                args_for_predict[arg] = kwargs[arg]
        self.results = self.predict(**args_for_predict)

        self.w = self.results["W"]
        self.h = self.results["H"]
        self.wl = self.results["WL"]
        self.hl = self.results["HL"]
        self.wdoth = self.w.dot(self.h.T)
        self.wdothl = self.wl.dot(self.hl.T)

        self.fig_result, self.fig_input, self.fig_diff, self.fig_result_l, self.fig_diff_l = (
            None, None, None, None, None
        )

    def fit_transform(
        self,
        w=None,
        h=None,
        update_h=True,
        update_w=True,
        n_bootstrap=None,
        regularization=None,
        sparsity=0,
        skewness=False,
        null_priors=False,
    ):

        """Compute Non-negative Matrix Factorization (NMF)

        Find two non-negative matrices (W, H) such as x = W @ H.T + Error.
        This factorization can be used for example for
        dimensionality reduction, source separation or topic extraction.

        The objective function is minimized with an alternating minimization of W
        and H.

        Parameters
        ----------

        w : np.ndarray, shape (n_samples, n_components)
            prior W
            If n_update_W == 0 , it is used as a constant, to solve for H only.

        h : np.ndarray, shape (n_features, n_components)
            prior H
            If n_update_H = 0 , it is used as a constant, to solve for W only.

        update_w : boolean, default: True
            Update or keep W fixed

        update_h : boolean, default: True
            Update or keep H fixed

        n_bootstrap : int, default: 0
            Number of bootstrap runs.

        regularization :  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.

        sparsity : float, default: 0
            Sparsity target with 0 <= sparsity <= 1 representing either:
            - the % rows in W or H set to 0 (when use_hals = False)
            - the mean % rows per column in W or H set to 0 (when use_hals = True)

        skewness : bool, default False
            When solving mixture problems, columns of X at the extremities of the convex hull will be given largest
            weights.
            The column weight is a function of the skewness and its sign.
            The expected sign of the skewness is based on the skewness of W components, as returned by the first pass
            of a 2-steps convex NMF. Thus, during the first pass, skewness must be set to False.
            Can be set only if convex = 'transformation' and prior W and H have been defined.

        null_priors : bool, default False
            Cells of H with prior cells = 0 will not be updated.
            Can be set only if prior H has been defined.

        Returns
        -------

        Estimator (dictionary) with following entries

        W : np.ndarray, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.

        H : np.ndarray, shape (n_components, n_features)
            Solution to the non-negative least squares problem.

        volume : scalar, volume occupied by W and H

        WB : np.ndarray, shape (n_samples, n_components)
            A sample is clustered in cluster k if its leverage on component k is higher than on any other components.
            During each run of the bootstrap, samples are re-clustered.
            Each row of WB contains the frequencies of the n_components clusters following the bootstrap.
                Only if n_bootstrap > 0.

        HB : np.ndarray, shape (n_components, n_features)
            A feature is clustered in cluster k if its leverage on component k is higher than on any other components.
            During each run of the bootstrap, features are re-clustered.
            Each row of HB contains the frequencies of the n_components clusters following the bootstrap.
                Only if n_bootstrap > 0.

        B : np.ndarray, shape (n_observations, n_components) or (n_features, n_components)
            Only if active convex variant, H = B.T @ X or W = X @ B

        diff : scalar, objective minimum achieved


        References
        ----------
        
        P. Fogel, D.M. Hawkins, C. Beecher, G. Luta, S. S. Young (2013). A Tale of Two Matrix Factorizations.
        The American Statistician, Vol. 67, Issue 4.

        C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
        IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1

        """

        return non_negative_factorization(
            self.x,
            W=w,
            H=h,
            n_components=self.n_components,
            update_W=update_w,
            update_H=update_h,
            beta_loss=self.beta_loss,
            use_hals=self.use_hals,
            n_bootstrap=n_bootstrap,
            tol=self.tol,
            max_iter=self.max_iter,
            max_iter_mult=self.max_iter_mult,
            regularization=regularization,
            sparsity=sparsity,
            leverage=self.leverage,
            convex=self.convex,
            kernel=self.kernel,
            skewness=skewness,
            null_priors=null_priors,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def predict(self, blocks=None, cluster_by_stability=False, custom_order=False):

        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------

        blocks : np.ndarray, shape(n_blocks), default None
            Size of each block (if any) in ordered heatmap.

        cluster_by_stability : bool, default False
             Use stability instead of leverage to assign samples/features to clusters

        custom_order :  bool, default False
             if False samples/features with highest leverage or stability appear on top of each cluster
             if True within cluster ordering is modified to suggest a continuum  between adjacent clusters

        Returns
        -------

        Completed estimator with following entries:
        WL : np.ndarray, shape (n_samples, n_components)
             Sample leverage on each component

        HL : np.ndarray, shape (n_features, n_components)
             Feature leverage on each component

        QL : np.ndarray, shape (n_blocks, n_components)
             Block leverage on each component (NTF only)

        WR : vector-like, shape (n_samples)
             Ranked sample indexes (by cluster and leverage or stability)
             Used to produce ordered heatmaps

        HR : vector-like, shape (n_features)
             Ranked feature indexes (by cluster and leverage or stability)
             Used to produce ordered heatmaps

        WN : vector-like, shape (n_components)
             Sample cluster bounds in ordered heatmap

        HN : vector-like, shape (n_components)
             Feature cluster bounds in ordered heatmap

        WC : vector-like, shape (n_samples)
             Sample assigned cluster

        HC : vector-like, shape (n_features)
             Feature assigned cluster

        QC : vector-like, shape (size(blocks))
             Block assigned cluster (NTF only)

        """

        return nmf_predict(
            self.estimator,
            blocks=blocks,
            leverage=self.leverage,
            cluster_by_stability=cluster_by_stability,
            custom_order=custom_order,
            verbose=self.verbose,
        )

    def permutation_test_score(self, estimator, y, n_permutations=100):

        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------

        estimator: dict
            Returned by fit_transform

        y :  np.ndarray
            Group to be predicted

        n_permutations:  int, default: 100

        Returns
        -------

        Completed estimator with following entries:

        score: float
             The true score without permuting targets.

        pvalue: float
             The p-value, which approximates the probability that the score would be obtained by chance.

        CS: np.ndarray, shape(n_components)
             The size of each cluster

        CP: np.ndarray, shape(n_components)
             The pvalue of the most significant group within each cluster

        CG: np.ndarray, shape(n_components)
             The index of the most significant group within each cluster

        CN: np.ndarray, shape(n_components, n_groups)
             The size of each group within each cluster

        """

        return nmf_permutation_test_score(estimator, y, n_permutations=n_permutations, verbose=self.verbose)

    def plot(self):

        self.fig_result = self.plot_matrices(
            "$W \\cdot H^t$",
            [self.wdoth, self.w, self.h.T],
            x_ax_labels=["features", "values"],
            w_ax_labels=[None, "values in metafeatures"],
            h_ax_labels=["metafeatures weights in features", None]
        )

        self.fig_result_l = self.plot_matrices(
            "$W_L \\cdot H_L^t$",
            [self.wdoth, self.wl, self.hl.T],
            x_ax_labels=["features", "values"],
            w_ax_labels=[None, "values in metafeatures"],
            h_ax_labels=["metafeatures weights in features", None]
        )

        self.fig_input = self.plot_matrices(
            "$X$",
            [self.x, self.w_input, self.h_input],
            x_ax_labels=["features", "values"],
            w_ax_labels=[None, "value in metafeatures"],
            h_ax_labels=["metafeature weights in feature", None]
        )

        self.fig_diff = self.plot_matrices(
            "$\\sqrt{\\frac{\\left(W \\cdot H^t - X\\right)^2}{X^2}}$ ($\\%$)",
            100 * np.sqrt(((self.wdoth - self.x)**2) / self.x**2)
        )

        self.fig_diff_l = self.plot_matrices(
            "$\\sqrt{\\frac{\\left(W_L \\cdot H_L^t - X\\right)^2}{X^2}}$ ($\\%$)",
            100 * np.sqrt(((self.wdothl - self.x)**2) / self.x**2)
        )

    def plot_matrices(
            self,
            text: str,
            matrices: Union[np.array, List[np.array]],
            x_ax_labels: List[str] = None,
            w_ax_labels: List[str] = None,
            h_ax_labels: List[str] = None,
    ) -> plt.Figure:

        if h_ax_labels is None:
            h_ax_labels = [None, None]
        if w_ax_labels is None:
            w_ax_labels = [None, None]
        if x_ax_labels is None:
            x_ax_labels = [None, None]

        if isinstance(matrices, np.ndarray):
            matrices = [matrices, None]
        if matrices[1] is None:
            fig, axes = plt.subplots()
            axes.set_title(text)
            self.plot_np(matrices[0], axes, x_ax_labels, r=3)
            return fig

        fig, axes = plt.subplots(2, 2)
        axes[0, 0].axis("off")
        axes[0, 0].text(0.5, 0.5, text, ha="center", va="center", fontsize=16)
        self.plot_np(matrices[0], axes[1, 1], x_ax_labels)
        if matrices[1] is not None:
            self.plot_np(matrices[1], axes[1, 0], w_ax_labels)
        else:
            axes[1, 0].axis("off")
        if matrices[2] is not None:
            self.plot_np(matrices[2], axes[0, 1], h_ax_labels)
        else:
            axes[0, 1].axis("off")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_np(arr, ax: plt.Axes, labels: List[str], r: int = 2) -> None:
        ax.imshow(arr, aspect="auto")
        ax.xaxis.set_label_position('top')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                ax.text(j, i, round(arr[i, j], r), ha="center", va="center", color="w", fontsize=8)


class NTF:
    """Initialize NTF model

    Parameters
    ----------
    n_components : int
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    fast_hals : bool, default: False
        Use fast implementation of HALS

    n_iter_hals : int, default: 2
        Number of HALS iterations prior to fast HALS

    n_shift : int, default: 0
        max shifting in convolutional NTF

    unimodal : Boolean, default: False

    smooth : Boolean, default: False

    apply_left : Boolean, default: False

    apply_right : Boolean, default: False

    apply_block : Boolean, default: False

    tol : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter : int, default: 200
        Maximum number of iterations.

    leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of W and H rows on each component.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, default: 0
        The verbosity level (0/1).


    Returns
    -------
    NTF model

    Example
    -------
    >>> from nmtf import *
    >>> myNTFmodel = NTF(n_components=4)

    Reference
    ---------
    A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

    """

    def __init__(
        self,
        n_components=None,
        fast_hals=False,
        n_iter_hals=2,
        n_shift=0,
        unimodal=False,
        smooth=False,
        apply_left=False,
        apply_right=False,
        apply_block=False,
        tol=1e-6,
        max_iter=150,
        leverage="standard",
        random_state=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.fast_hals = fast_hals
        self.n_iter_hals = n_iter_hals
        self.n_shift = n_shift
        self.unimodal = unimodal
        self.smooth = smooth
        self.apply_left = apply_left
        self.apply_right = apply_right
        self.apply_block = apply_block
        self.tol = tol
        self.max_iter = max_iter
        self.leverage = leverage
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(
        self,
        x: np.ndarray,
        n_blocks: int,
        n_bootstrap: int = None,
        regularization: str = None,
        sparsity: float = 0,
        w: Union[np.ndarray, None] = None,
        h: Union[np.ndarray, None] = None,
        q: Union[np.ndarray, None] = None,
        update_w: Union[np.ndarray, None] = True,
        update_h: Union[np.ndarray, None] = True,
        update_q: Union[np.ndarray, None] = True,
    ):

        """Compute Non-negative Tensor Factorization (NTF)

        Find three non-negative matrices (W, H, Q) such as x = W @@ H @@ Q + Error (@@ = tensor product).
        This factorization can be used for example for
        dimensionality reduction, source separation or topic extraction.

        The objective function is minimized with an alternating minimization of W
        and H.

        Parameters
        ----------

        x: np.ndarray, shape (n_samples, n_features x n_blocks)
            Constant matrix.
            X is a tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.

        n_blocks: int
            Number of blocks defining the 3rd dimension of the tensor

        n_bootstrap: int
            Number of bootstrap runs

        regularization: str  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.

        sparsity: float, default: 0
            Sparsity target with 0 <= sparsity <= 1 representing the mean % rows per column in W or H set to 0
.
        w: Union[np.ndarray, None], shape (n_samples, n_components)
            prior W

        h: Union[np.ndarray, None], shape (n_features, n_components)
            prior H

        q: Union[np.ndarray, None], shape (n_blocks, n_components)
            prior Q

        update_w: bool, default: True
            Update or keep w fixed

        update_h: bool, default: True
            Update or keep h fixed

        update_q: bool, default: True
            Update or keep q fixed
        
        Returns
        -------

        Estimator (dictionary) with following entries

        W : np.ndarray, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.

        H : np.ndarray, shape (n_features, n_components)
            Solution to the non-negative least squares problem.

        Q : np.ndarray, shape (n_blocks, n_components)
            Solution to the non-negative least squares problem.
        
        volume : scalar, volume occupied by W and H

        WB : np.ndarray, shape (n_samples, n_components)
            Percent consistently clustered rows for each component.
            only if n_bootstrap > 0.

        HB : np.ndarray, shape (n_features, n_components)
            Percent consistently clustered columns for each component.
            only if n_bootstrap > 0.
        
        diff : scalar, objective minimum achieved

        Example
        -------
        >>> from nmtf import *
        >>> myNTFmodel = NTF(n_components=4)
        >>> # M: tensor with 5 blocks to be factorized
        >>> estimator = myNTFmodel.fit_transform(M, 5)
        
        Reference
        ---------

        A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

        """

        return non_negative_tensor_factorization(
            x,
            n_blocks,
            W=w,
            H=h,
            Q=q,
            n_components=self.n_components,
            update_W=update_w,
            update_H=update_h,
            update_Q=update_q,
            fast_hals=self.fast_hals,
            n_iter_hals=self.n_iter_hals,
            n_shift=self.n_shift,
            regularization=regularization,
            sparsity=sparsity,
            unimodal=self.unimodal,
            smooth=self.smooth,
            apply_left=self.apply_left,
            apply_right=self.apply_right,
            apply_block=self.apply_block,
            n_bootstrap=n_bootstrap,
            tol=self.tol,
            max_iter=self.max_iter,
            leverage=self.leverage,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def predict(self, estimator, blocks=None, cluster_by_stability=False, custom_order=False):

        """See function description in class NMF

        """

        return nmf_predict(
            estimator,
            blocks=blocks,
            leverage=self.leverage,
            cluster_by_stability=cluster_by_stability,
            custom_order=custom_order,
            verbose=self.verbose,
        )

    @staticmethod
    def permutation_test_score(estimator, y, n_permutations=100):

        """See function description in class NMF

        """

        return nmf_permutation_test_score(estimator, y, n_permutations=n_permutations)
