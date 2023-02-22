import pandas as pd

from hybridfunction import HybridFunction


def test_custom_function():
    """ test function behaves as expected:
    values less than x0 -> y0, values greater-equal x0 -> y = mx + c   (eqn of straight line)
        m is the gradient, c is the intercept = y0 - (gradient * x0)
    """
    # instance of the model function with x0, y0, gradient set
    model = HybridFunction(x0=5, y0=2, gradient=1)
    # test data in a Pandas series -2 ... 12
    data = pd.DataFrame([range(-2, 13)])

    # apply the model's _custom_function to get output in a Pandas series
    model_output = data.apply(model._custom_function, axis=1)

    assert int(model_output[0]) == 2
    assert int(model_output[7]) == 2
    assert int(model_output[12]) == 7

def test_predict_method():
    """ test function behaves as expected:
    values less than x0 -> y0, values greater-equal x0 -> y = mx + c   (eqn of straight line)
        m is the gradient, c is the intercept = y0 - (gradient * x0)
    """
    # instance of the model function with x0, y0, gradient set
    model = HybridFunction(x0=5, y0=2, gradient=1)
    # test data in a Pandas series -2 ... 12
    data = pd.DataFrame([range(-2, 13)])

    # apply the model's _custom_function to get output in a Pandas series
    model_output = model.predict(context=None, model_input=data)

    assert int(model_output[0]) == 2
    assert int(model_output[7]) == 2
    assert int(model_output[12]) == 7
