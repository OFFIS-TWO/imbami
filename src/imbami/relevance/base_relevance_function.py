import numpy as np
from abc import ABC, abstractmethod
from ..core.helpers import bisect_kde_score
from ..core.density_estimation import get_kernel
from typing import Literal, Callable



class RelevanceFunctionBase(ABC):
    """
    Universal interface that ALL relevance functions must implement.
    """
    def __init__(self) -> None:
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray, *args, **kwargs) -> None:
        """
        Fit the relevance function to data.
        
        Args:
            **fit_params: Parameters needed for fitting (varies by implementation)
            
        Returns:
            self (for method chaining)
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self, y: np.ndarray) -> np.ndarray:
        """
        Compute relevance scores for input values.

        Must be implemented by subclasses to define the specific relevance computation.

        Args:
            y: Input values for which to compute relevance scores.

        Returns:
            Computed relevance scores for the input values.
        """
        raise NotImplementedError

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """
        Compute relevance scores by calling eval().

        Args:
            y: Input values for which to compute relevance scores.

        Returns:
            Computed relevance scores for the input values.
        """
        return self.eval(y)












class EmpiricalDomainRelevanceFunctionBase(RelevanceFunctionBase, ABC):
    """
    Abstract base class for relevance functions that compare empirical and domain densities.

    Provides shared functionality for empirical and domain density estimation using either
    Kernel Density Estimation (KDE) or provided probability density functions (PDFs).
    Subclasses must implement the `eval()` method to compute relevance scores.

    Attributes:
        emp_density_mode (Literal['fit_kde', 'provide_pdf']): Mode for empirical density estimation.
        domain_density_mode (Literal['fit_kde', 'provide_pdf']): Mode for domain density estimation.
        is_fitted (bool): Whether the relevance function has been fitted to data.
        usable_bw_type (list): Supported bandwidth types for KDE ('silverman', 'ISJ', 'uniform').
    """
    def __init__(self,
                 emp_density_mode: Literal['fit_kde', 'provide_pdf'],
                 domain_density_mode: Literal['fit_kde', 'provide_pdf']):
        """
        Initialize the BaseRelevanceFunction.

        Args:
            emp_density_mode: Method for empirical density estimation ('fit_kde' or 'provide_pdf').
            domain_density_mode: Method for domain density estimation ('fit_kde' or 'provide_pdf').

        Note:
            This is an abstract base class and cannot be instantiated directly.
            Use a concrete subclass like DensityRatioRelevance instead.
        """
        super().__init__()
        self.usable_bw_type = ['silverman', 'ISJ', 'uniform']
        self._bisect_kde_score = bisect_kde_score
        self.get_kernel = get_kernel

        self.emp_density_mode = emp_density_mode
        self.domain_density_mode = domain_density_mode

        

    def fit(self, *args, **kwargs) -> None:
        """
        Base fit just marks object as fitted.
        Subclasses extend this.
        """
        self.is_fitted = True

    def _fit_to_data(self,
            emp_data: None | np.ndarray,
            emp_bw_type: None | str = 'silverman',
            emp_bw_factor: None | float = 1.0,
            emp_kernel_type: None | str = 'gaussian',
            emp_pdf: None | Callable = None,

            domain_data: None | np.ndarray = None,
            domain_bw_type: None | str = 'uniform',
            domain_bw_factor: None | float = 1.0,
            domain_kernel_type: None | str = 'gaussian',
            domain_pdf: None | Callable = None) -> None:
        """
        Fit density estimators to empirical and domain data.

        Configures either KDEs or provided PDFs for both empirical and domain distributions
        based on the specified density modes.

        Args:
            emp_data: Empirical data samples. Required if emp_density_mode is 'fit_kde'.
            emp_bw_type: Bandwidth selection method for empirical KDE ('silverman', 'ISJ', or 'uniform').
            emp_bw_factor: Scaling factor for empirical KDE bandwidth.
            emp_kernel_type: Kernel type for empirical KDE (e.g., 'gaussian').
            emp_pdf: Callable PDF for empirical distribution. Required if emp_density_mode is 'provide_pdf'.

            domain_data: Domain data samples. Required if domain_density_mode is 'fit_kde'.
            domain_bw_type: Bandwidth selection method for domain KDE ('silverman', 'ISJ', or 'uniform').
            domain_bw_factor: Scaling factor for domain KDE bandwidth.
            domain_kernel_type: Kernel type for domain KDE (e.g., 'gaussian').
            domain_pdf: Callable PDF for domain distribution. Required if domain_density_mode is 'provide_pdf'.

        Raises:
            ValueError: If required arguments are missing or invalid for the specified density modes.
        """
        # Empirical
        match self.emp_density_mode:
            case 'fit_kde':
                if emp_data is None:
                    raise ValueError("emp_data is None. It must be provided if empirical density mode is set to 'fit_kde'.")
                if emp_bw_type is None:
                    raise ValueError("emp_bw_type is None. It must be provided if empirical density mode is set to 'fit_kde'.")
                if emp_bw_factor is None:
                    raise ValueError("emp_bw_factor is None. It must be provided if empirical density mode is set to 'fit_kde'.")
                if emp_kernel_type is None:
                    raise ValueError("emp_kernel_type is None. It must be provided if empirical density mode is set to 'fit_kde'.")
                
                self.emp_data = emp_data
                self.emp_bw_type = emp_bw_type
                self.emp_kernel_type = emp_kernel_type
                self.emp_kernel = self.get_kernel(data = self.emp_data,
                                                bw_type= self.emp_bw_type,
                                                bw_factor= emp_bw_factor,
                                                kernel_type= self.emp_kernel_type)
                self.emp_grid_x, self.emp_grid_y = self.emp_kernel.evaluate(2**12) # 2^12 = 4096 grid points
            case 'provide_pdf':
                if emp_pdf is None:
                    raise ValueError("emp_pdf is None. It must be provided if empirical density mode is set to 'provide_pdf'.")
                if isinstance(emp_pdf, Callable):
                    self.emp_pdf = emp_pdf
                else:
                    raise ValueError("emp_pdf is no Callable. It must be if empirical density mode is set to 'provide_pdf'.")
                    
            case _:
                raise ValueError(f"empirical_density_mode has to be either 'fit_kde' or 'provide_pdf'.")

        # Domain
        match self.domain_density_mode:
            case 'fit_kde':
                if domain_data is None:
                    raise ValueError("domain_data is None. It must be provided if domain density mode is set to 'fit_kde'.")
                if domain_bw_type is None:
                    raise ValueError("domain_bw_type is None. It must be provided if domain density mode is set to 'fit_kde'.")
                if domain_bw_factor is None:
                    raise ValueError("domain_bw_factor is None. It must be provided if domain density mode is set to 'fit_kde'.")
                if domain_kernel_type is None:
                    raise ValueError("domain_kernel_type is None. It must be provided if domain density mode is set to 'fit_kde'.")
                
                self.domain_data = domain_data
                self.domain_bw_type = domain_bw_type
                self.domain_kernel_type = domain_kernel_type
                self.domain_kernel = self.get_kernel(data = self.domain_data,
                                                bw_type= self.domain_bw_type,
                                                bw_factor= domain_bw_factor,
                                                kernel_type= self.domain_kernel_type)
                self.domain_grid_x, self.domain_grid_y = self.domain_kernel.evaluate(2**12) # 2^12 = 4096 grid points
            case 'provide_pdf':
                if domain_pdf is None:
                    raise ValueError("domain_pdf is None. It must be provided if domain density mode is set to 'provide_pdf'.")
                if isinstance(domain_pdf, Callable):
                    self.domain_pdf = domain_pdf
                else:
                    raise ValueError("domain_pdf is no Callable. It must be if domain density mode is set to 'provide_pdf'.")
            case _:
                raise ValueError(f"domain_density_mode has to be either 'fit_kde' or 'provide_pdf'.")
        
        
        self.is_fitted = True


    def _get_densities_from_grid(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute density values for input points from both empirical and domain distributions.

        Args:
            y: Input values for which to compute densities.

        Returns:
            Tuple containing:
                - emp_density: Density values from empirical distribution
                - domain_density: Density values from domain distribution
        """
        if y is None or len(y) == 0:
            raise ValueError("Input y cannot be None or empty.")
        if not isinstance(y, np.ndarray):
            raise TypeError("Input y must be a numpy array.")
        if np.any(~np.isfinite(y)):
            raise ValueError("Input y contains NaN or infinite values.")
        if self.emp_density_mode == 'fit_kde':
            if not hasattr(self, 'emp_grid_x') or not hasattr(self, 'emp_grid_y'):
                raise RuntimeError("Empirical KDE not properly initialized.")
            emp_density = np.array([self._bisect_kde_score(i, self.emp_grid_x, self.emp_grid_y) for i in y])
        else:
            emp_density = self.emp_pdf(y)

        if self.domain_density_mode == 'fit_kde':
            if not hasattr(self, 'domain_grid_x') or not hasattr(self, 'domain_grid_y'):
                raise RuntimeError("Domain KDE not properly initialized.")
            domain_density = np.array([self._bisect_kde_score(i, self.domain_grid_x, self.domain_grid_y) for i in y])
        else:
            domain_density = self.domain_pdf(y)
        return emp_density, domain_density



    





    
