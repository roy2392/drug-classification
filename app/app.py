import gradio as gr
import skops.io as sio

unknown_types = sio.get_untrusted_types(file="./Model/drug_pipeline.skops")
pipe = sio.load("./Model/drug_pipeline.skops", trusted=unknown_types)


def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features.

    :param age: Age of patient
    :type age: int
    :param sex: Sex of patient
    :type sex: str
    :param blood_pressure: Blood pressure level
    :type blood_pressure: str
    :param cholesterol: Cholesterol level
    :type cholesterol: str
    :param na_to_k_ratio: Ratio of sodium to potassium in blood
    :type na_to_k_ratio: float
    :returns: Predicted drug label
    :rtype: str

    """
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]
    predicted_drug = pipe.predict([features])[0]

    label = f"Predicted Drug: {predicted_drug}"
    return label


inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]
outputs = [gr.Label(num_top_classes=5)]

examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]


title = "Drug Classification"
description = "Enter the details to correctly identify Drug type?"
article = "This app is a part of the **[Beginner's Guide to CI/CD for Machine Learning](https://www.datacamp.com/tutorial/ci-cd-for-machine-learning)**. It teaches how to automate training, evaluation, and deployment of models to Hugging Face using GitHub Actions."


gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()
