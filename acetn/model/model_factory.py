from .model import Model
from .models import models

class ModelFactory:
    """
    A factory class for creating and managing model instances.

    The `ModelFactory` class allows for the registration of different model classes and the creation of 
    model instances based on a string identifier. It ensures that models are properly registered and 
    instantiated using the provided configuration.

    Attributes:
        _models (dict): A dictionary mapping model names (strings) to their corresponding model classes.
    """
    def __init__(self):
        """
        Initializes the `ModelFactory` with a predefined set of model classes.

        The constructor initializes the `_models` attribute with a set of model classes, such as 
        'ising' and 'heisenberg'. These models can later be instantiated using the `create` method.

        Returns:
            None
        """
        self._models = models

    def register(self, name, cls):
        """
        Registers a new model class under a given name.

        This method allows users to register new model classes with the factory. The registered class 
        must inherit from the `Model` base class.

        Args:
            name (str): The name associated with the model class.
            cls (type): The model class that is to be registered.

        Raises:
            TypeError: If the provided class does not inherit from the `Model` base class.

        Returns:
            None
        """
        if not issubclass(cls, Model):
            raise TypeError("Model class must inherit from Model base class")
        self._models[name] = cls

    def create(self, config):
        """
        Creates an instance of a model based on the registered model name.

        This method instantiates a model class that has been registered under the specified `name` and 
        initializes it with the provided configuration.

        Args:
            name (str): The name of the registered model.
            config (dict): A configuration dictionary to initialize the model.

        Raises:
            KeyError: If the provided model name is not registered.
        
        Returns:
            Model: An instance of the registered model class initialized with the provided configuration.
        """
        if config.name is None:
            raise ValueError("Model has not been set.")
        elif config.name not in self._models:
            raise KeyError(f"Model name: '{config.name}' is not registered.")
        else:
            return self._models[config.name](config)

    def list_registered(self):
        """
        Lists all the registered model names.

        This method returns a list of all model names that have been registered with the factory.

        Returns:
            list: A list of strings representing the names of all registered models.
        """
        return list(self._models.keys())

model_factory = ModelFactory()
