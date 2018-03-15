from sklearn.base import BaseEstimator, TransformerMixin

class FunctionFilter(BaseEstimator, TransformerMixin):
    """
    A wrapper for filters already available in common image libraries. The signature of those functions must start with
    the image parameter though for this class to be the suitable one.
    Parameters:
    -----------
        filter_function: function
        The filter function wrapped into this class, like gabor, gaussian, etc.
        pargs: Variable length positional parameters
        kwarg: Variable lenght keyword parameters
    """
    def __init__(self, filter_function, *pargs, **kwargs):
        self.filter_function = filter_function
        self.pargs = pargs
        self.kwargs = kwargs

    def fit(self):
        """
        No parameter is learnt by the filter function from the input image so this method is provided just for
        compatibility reasons but it just returns self for chaining convenience.
        """
        return self

    def transform(self, image):
        """
        Transform the input image by applying the filter function provided in the constructor.

        Parameters:
        -----------
            image: image
        """
        return self.filter_function(image, *self.pargs, **self.kwargs)
