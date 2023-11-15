import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnorstics import performance_metrics
import mlflow
import mlflow.pyfunc

class FbProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def load_context(self, context):
        from fbprophetport Prophet
        return

    def predict(self, context, model_input):
        future = self.model.make_future_dataframe(periods=model_input["periods"][0])

        return self.model.predict(future)

with mlflow.start_run():

# create Prophet model
model = Prophet(
    yearly_seasonality=seasonality_params['yearly']
    weekly_seasonality=seasonality_params['weekly']
    daily_seasonality=seasonality_params['daily']
)


# train and predioct
model.fit(df_train)

# Evaluate metrics
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='360 days')
df_p = performance_metrics(dfffffffff_cv)

mlflow.log_metric("rmse", df_p.loc[0, "rmse"])


mlflow.pyfunc.log_model("model", python_model=FbProphetWrapper(model))
print(
    "Logged model with URI: runs:/{run_id}/model".format(
        run_id=mlflow.active_run().info.run_id
    )
)





