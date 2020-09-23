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


class NMTF:
    """Abstract class from which NMF and NTF derive"""

    def __init__(
        self,
        which: str,
        x: Union[np.ndarray, None] = None,
        w_input: Union[np.ndarray, None] = None,
        h_input: Union[np.ndarray, None] = None,
        q_input: Union[np.ndarray, None] = None,
        n_blocks: int = None,
        n_permutations: int = 100,
        blocks: np.ndarray = None,
        cluster_by_stability: bool = False,
        custom_order: bool = False,
        wp: Union[np.ndarray, None] = None,
        hp: Union[np.ndarray, None] = None,
        qp: Union[np.ndarray, None] = None,
        n_components: int = None,
        update_h: bool = True,
        update_w: bool = True,
        update_q: bool = True,
        beta_loss: str = 'frobenius',
        tol: float = 1e-6,
        max_iter: int = 200,
        max_iter_mult: int = 20,
        convex: str = None,
        kernel: str = "linear",
        random_state: Union[int, np.random.RandomState, None] = None,
        n_bootstrap: int = 0,
        regularization: str = None,
        sparsity: float = 0,
        use_hals: bool = False,
        fast_hals: bool = False,
        n_iter_hals: int = 2,
        n_shift: int = 0,
        unimodal: bool = False,
        smooth: bool = False,
        apply_left: bool = False,
        apply_right: bool = False,
        apply_block: bool = False,
        skewness: bool = False,
        null_priors: bool = False,
        leverage: str = "standard",
        verbose: int = 0,
    ):
        """Initialize NMF model

        Parameters
        ----------

        which: str
            "NMF" or "NTF"

        x: Union[np.ndarray, None]
            The x matrix to factorize, Can not be specified if w or h or q was spacified

        w_input: Union[np.ndarray, None]
            The cluster matrix from which to generate x. Can not be specified if x was specified.
            Need to specify also h (and q if which is "NTF")

        h_input: Union[np.ndarray, None]
            The weight matrix from which to generate x. Can not be specified if x was specified.
            Need to specify also w (and q if which is "NTF")

        q_input: Union[np.ndarray, None]
            The 3D dimension matrix from which to generate x. Can not be specified if x was specified.
            Need to specify also w and h.
            To manage this third dimension, x will be flatten by doing x . (h + q), were '+' denotes the concatenation
            of h and q.

        n_blocks: int, default None
            Number of blocks defining the 3rd dimension of the tensor (Mandatory if which = "NTF")

        n_permutations:  int, default: 100

        blocks : np.ndarray, shape(n_blocks), default None
            Size of each block (if any) in ordered heatmap.

        cluster_by_stability : bool, default False
             Use stability instead of leverage to assign samples/features to clusters

        custom_order :  bool, default False
             if False samples/features with highest leverage or stability appear on top of each cluster
             if True within cluster ordering is modified to suggest a continuum  between adjacent clusters

        wp: np.ndarray, shape (n_samples, n_components)
            prior w
            If n_update_w == 0 , it is used as a constant, to solve for h only.

        hp: np.ndarray, shape (n_features, n_components)
            prior h
            If n_update_h = 0 , it is used as a constant, to solve for w only.

        qp: np.ndarray, shape (n_features, n_components)
            prior q
            If n_update_q = 0 , it is used as a constant, to solve for q only.

        n_components: int, default = None
            Number of components, if n_components is not set : n_components = min(n_samples, n_features)

        update_w: boolean, default: True
            Update or keep w fixed

        update_h: boolean, default: True
            Update or keep h fixed

        update_q: boolean, default: True
            Update or keep q fixed

        beta_loss: str, default 'frobenius'
            String must be in {'frobenius', 'kullback-leibler'}.
            Beta divergence to be minimized, measuring the distance between x
            and the dot product WH. Note that values different from 'frobenius'
            (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
            fits. Note that for beta_loss == 'kullback-leibler', the input
            matrix x cannot contain zeros.

        tol: float, default: 1e-6
            Tolerance of the stopping condition.

        max_iter: int, default: 200
            Maximum number of iterations.

        max_iter_mult: int, default: 20
            Maximum number of iterations in multiplicative warm-up to projected gradient (beta_loss = 'frobenius' only).

        convex:  str, None | 'components' | 'transformation', default None
            Apply convex constraint on W or H.

        kernel: str, 'linear' | 'quadratic' | 'radial', default 'linear'
            Can be set if convex = 'transformation'.

        random_state: Union[int, np.random.RandomState, None], default: None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        n_bootstrap: int, default: 0
            Number of bootstrap runs.

        regularization:  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.

        sparsity: float, default: 0
            Sparsity target with 0 <= sparsity <= 1 representing either:
            - the % rows in W or H set to 0 (when use_hals = False)
            - the mean % rows per column in W or H set to 0 (when use_hals = True)

        use_hals: bool, default: False
            True -> HALS algorithm (note that convex & kullback-leibler loss options are not supported)
            False-> Projected gradiant

        fast_hals: bool, default: False
            Use fast implementation of HALS

        n_iter_hals: int, default: 2
            Number of HALS iterations prior to fast HALS

        n_shift: int, default: 0
            max shifting in convolutional NTF

        unimodal: bool, default: False

        smooth: bool, default: False

        apply_left: bool, default: False

        apply_right: bool, default: False

        apply_block: bool, default: False

        skewness: bool, default False
            When solving mixture problems, columns of X at the extremities of the convex hull will be given largest
            weights.
            The column weight is a function of the skewness and its sign.
            The expected sign of the skewness is based on the skewness of W components, as returned by the first pass
            of a 2-steps convex NMF. Thus, during the first pass, skewness must be set to False.
            Can be set only if convex = 'transformation' and prior W and H have been defined.

        null_priors: bool, default False
            Cells of H with prior cells = 0 will not be updated.
            Can be set only if prior H has been defined.

        leverage: str, None | 'standard' | 'robust', default 'standard'
            Calculate leverage of W and H rows on each component.

        verbose: int, default 0

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

        self.which = which.upper()  # 'nmf' or 'ntf' will also work
        self.x = x
        self.w_input = w_input
        self.h_input = h_input
        self.q_input = q_input
        self.n_blocks = n_blocks
        self.n_permutations = n_permutations
        self.blocks = blocks
        self.cluster_by_stability = cluster_by_stability
        self.custom_order = custom_order
        self.wp = wp
        self.hp = hp
        self.qp = qp
        self.n_components = n_components
        self.update_h = update_h
        self.update_w = update_w
        self.update_q = update_q
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.max_iter_mult = max_iter_mult
        self.convex = convex
        self.kernel = kernel
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        self.regularization = regularization
        self.sparsity = sparsity
        self.use_hals = use_hals
        self.fast_hals = fast_hals
        self.n_iter_hals = n_iter_hals
        self.n_shift = n_shift
        self.unimodal = unimodal
        self.smooth = smooth
        self.apply_left = apply_left
        self.apply_right = apply_right
        self.apply_block = apply_block
        self.skewness = skewness
        self.null_priors = null_priors
        self.leverage = leverage
        self.verbose = verbose

        self.x_index_name = "Values"
        self.x_cols_name = "Features"

        # Test arguments

        if self.which == "NMF":
            if h_input is None and w_input is not None:
                raise ValueError("You specified w but not h. Need both!")
            elif w_input is None and h_input is not None:
                raise ValueError("You specified h but not w. Need both!")
            elif x is None and w_input is None:  # if w is None so is h, otherwise would have raised already
                raise ValueError("Need x or w and h!")
            elif x is not None and w_input is not None:  # if w is not None neither is h, otherwise would have raised
                raise ValueError("Can not specify both x and w/h!")
            elif w_input is not None and h_input is not None:
                try:
                    self.x = w_input.dot(h_input)
                except ValueError:
                    self.x = w_input.dot(h_input.T)

        elif self.which == "NTF":

            if self.n_blocks is None:
                raise ValueError("If using NTF, must specify n_blocks")
            test_number = sum([w_input is None, h_input is None, q_input is None])
            if test_number == 3 and x is None:  # no x, w, h, or q was specified
                raise ValueError("You need to specify either x or w, h and q.")
            elif 0 <= test_number < 3 and x is not None:  # x was specified but also some of w, h and q
                raise ValueError("You can not specify both x and w, h and/or q.")
            elif 0 < test_number < 3 and x is None:  # some but not all of w, h and q was specified but not x
                raise ValueError("You specified only some of w, h and q but I need the three of them!")
            elif test_number == 0:  # w, h, and q were specified : compute x from them
                try:
                    self.x = w_input.dot(np.concatenate([h_input, q_input], axis=1))
                except ValueError:
                    self.x = w_input.dot(np.concatenate([h_input, q_input], axis=0).T)
            elif x is None:
                raise ValueError(f"Something weird happened. Check x, w, h and q:\n"
                                 f" - x: {x}\n - w: {w_input}\n - h: {h_input}\n - q: {q_input}")
        else:
            raise ValueError(f"Argument 'which' can only be 'NMF' or 'NTF', got '{self.which}'.")

        self.b = self.x_approx = self.x_approx_l = np.array([])
        self.w = self.h = self.q = self.wb = self.hb = np.array([])
        self.wl = self.hl = self.ql = self.hr = self.qr = np.array([])
        self.wn = self.hn = self.wc = self.hc = self.qc = np.array([])
        self.volume = self.diff = 0

        self.estimator = None
        self.results = None

        self.fig_result, self.fig_input, self.fig_diff, self.fig_result_l, self.fig_diff_l = (
            None, None, None, None, None
        )

    def fit_transform(self):

        """Compute Non-negative Matrix Factorization (NMF)

        Find two non-negative matrices (W, H) such as x = W @ H.T + Error.
        This factorization can be used for example for
        dimensionality reduction, source separation or topic extraction.

        The objective function is minimized with an alternating minimization of W
        and H.

        Returns
        -------

        Estimator (dictionary) with following entries

        w : np.ndarray, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.

        h : np.ndarray, shape (n_components, n_features)
            Solution to the non-negative least squares problem.

        q : np.ndarray, shape (n_blocks, n_components)
            Solution to the non-negative least squares problem. (if which is 'NTF')

        volume : scalar, volume occupied by W and H

        wb : np.ndarray, shape (n_samples, n_components)
            A sample is clustered in cluster k if its leverage on component k is higher than on any other components.
            During each run of the bootstrap, samples are re-clustered.
            Each row of WB contains the frequencies of the n_components clusters following the bootstrap.
                Only if n_bootstrap > 0.

        hb : np.ndarray, shape (n_components, n_features)
            A feature is clustered in cluster k if its leverage on component k is higher than on any other components.
            During each run of the bootstrap, features are re-clustered.
            Each row of HB contains the frequencies of the n_components clusters following the bootstrap.
                Only if n_bootstrap > 0.

        b : np.ndarray, shape (n_observations, n_components) or (n_features, n_components)
            Only if active convex variant, H = B.T @ X or W = X @ B

        diff : scalar, objective minimum achieved


        References
        ----------
        
        P. Fogel, D.M. Hawkins, C. Beecher, G. Luta, S. S. Young (2013). A Tale of Two Matrix Factorizations.
        The American Statistician, Vol. 67, Issue 4.

        C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
        IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1

        """

        if self.which == "NMF":
            self.estimator = non_negative_factorization(
                self.x,
                W=self.wp,
                H=self.hp,
                n_components=self.n_components,
                update_W=self.update_w,
                update_H=self.update_h,
                beta_loss=self.beta_loss,
                use_hals=self.use_hals,
                n_bootstrap=self.n_bootstrap,
                tol=self.tol,
                max_iter=self.max_iter,
                max_iter_mult=self.max_iter_mult,
                regularization=self.regularization,
                sparsity=self.sparsity,
                leverage=self.leverage,
                convex=self.convex,
                kernel=self.kernel,
                skewness=self.skewness,
                null_priors=self.null_priors,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        else:
            self.estimator = non_negative_tensor_factorization(
                self.x,
                self.n_blocks,
                W=self.wp,
                H=self.hp,
                Q=self.qp,
                n_components=self.n_components,
                update_W=self.update_w,
                update_H=self.update_h,
                update_Q=self.update_q,
                fast_hals=self.fast_hals,
                n_iter_hals=self.n_iter_hals,
                n_shift=self.n_shift,
                regularization=self.regularization,
                sparsity=self.sparsity,
                unimodal=self.unimodal,
                smooth=self.smooth,
                apply_left=self.apply_left,
                apply_right=self.apply_right,
                apply_block=self.apply_block,
                n_bootstrap=self.n_bootstrap,
                tol=self.tol,
                max_iter=self.max_iter,
                leverage=self.leverage,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        for arg in self.estimator:
            setattr(self, arg, self.estimator[arg])

    def predict(self):

        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps"""

        self.results = nmf_predict(
            self.estimator,
            blocks=self.blocks,
            leverage=self.leverage,
            cluster_by_stability=self.cluster_by_stability,
            custom_order=self.custom_order,
            verbose=self.verbose,
        )
        for arg in self.results:
            setattr(self, arg, self.results[arg])

        if self.which == "NMF":
            self.x_approx = self.w.dot(self.h.T)
            self.x_approx_l = self.wl.dot(self.hl.T)
        else:
            self.x_approx = self.w.dot(np.concatenate([self.h, self.q], axis=0).T)
            self.x_approx_l = self.wl.dot(np.concatenate([self.hl, self.q], axis=0).T)

    def permutation_test_score(self, y: np.ndarray):

        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------

        y :  np.ndarray
            Group to be predicted

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

        return nmf_permutation_test_score(self.estimator, y, n_permutations=self.n_permutations, verbose=self.verbose)

    def plot(self):

        self.fig_result = self.plot_matrices(
            "$W \\cdot H^t$",
            [self.x_approx, self.w, self.h.T],
            x_ax_labels=["features", "values"],
            w_ax_labels=[None, "values in metafeatures"],
            h_ax_labels=["metafeatures weights in features", None]
        )

        self.fig_result_l = self.plot_matrices(
            "$W_L \\cdot H_L^t$",
            [self.x_approx, self.wl, self.hl.T],
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
            100 * np.sqrt(((self.x_approx - self.x) ** 2) / self.x ** 2)
        )

        self.fig_diff_l = self.plot_matrices(
            "$\\sqrt{\\frac{\\left(W_L \\cdot H_L^t - X\\right)^2}{X^2}}$ ($\\%$)",
            100 * np.sqrt(((self.x_approx_l - self.x) ** 2) / self.x ** 2)
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


class NMF(NMTF):

    def __init__(
        self,
        **kwargs
    ):
        NMTF.__init__(self, which="NMF", **kwargs)


class NTF(NMTF):

    def __init__(
        self,
        **kwargs
    ):
        NMTF.__init__(self, which="NTF", **kwargs)
