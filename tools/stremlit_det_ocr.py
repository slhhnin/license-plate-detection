import streamlit as st
import cv2
from PIL import Image
import numpy as np
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
import os
import time
from loguru import logger
import torch
from yolox.data.datasets import COCO_CLASSES
from demo import Predictor


def image_demo(predictor, vis_folder, image, current_time, img_name, save_result):

    outputs, img_info = predictor.inference(image)
    result_image, count, output_text = predictor.visual(
        outputs[0], img_info, predictor.confthre
    )

    if save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        save_file_name = os.path.join(save_folder, img_name)
        logger.info("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, result_image)

    return result_image, output_text


def main(exp, args):

    args["experiment_name"] = exp.exp_name

    file_name = os.path.join(exp.output_dir, args["experiment_name"])
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args["save_result"]:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args["trt"]:
        args["device"] = "gpu"

    logger.info("Args: {}".format(args))

    if args["conf"] is not None:
        exp.test_conf = args["conf"]
    if args["nms"] is not None:
        exp.nmsthre = args["nms"]
    if args["tsize"] is not None:
        exp.test_size = (args["tsize"], args["tsize"])

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args["device"] == "gpu":
        model.cuda()
        if args["fp16"]:
            model.half()  # to FP16
    model.eval()

    if not args["trt"]:

        if args["ckpt"] is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args["ckpt"]

        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args["fuse"]:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args["trt"]:
        assert not args["fuse"], "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model,
        exp,
        COCO_CLASSES,
        trt_file,
        decoder,
        args["device"],
        args["fp16"],
        args["legacy"],
    )
    current_time = time.localtime()
    # if args.demo == "image":
    result_image, ocr_result = image_demo(
        predictor,
        vis_folder,
        args["path"],
        current_time,
        args["img_name"],
        args["save_result"],
    )
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    # Create table content

    # Streamlit cannot display images directly in a table, so display them in columns
    st.subheader("Inference Results")

    # Create three columns to display results side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(result_image, caption="Detected Image")
    with col2:
        st.write("**OCR Result:** \n")
        st.markdown(
            f"<span style='background-color: rgb(255, 159, 51);'>{ocr_result}</span>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":

    # Streamlit UI
    st.title("Inference and OCR Viewer")

    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")
    model_name = st.sidebar.text_input("Model Name", value="yolox-s")

    # Experiment file and checkpoint
    exp_file = st.sidebar.text_input(
        "Experiment File (exp_file)", value="exps/default/yolox_s.py"
    )

    checkpoint = st.file_uploader("Upload a checkpoint file", type=[".pth"])
    # Check if checkpoint is empty
    if not checkpoint:
        # checkpoint = "YOLOX_outputs/yolox_s/latest_ckpt.pth"
        st.error("Checkpoint is required. Please provide a value.")
        st.stop()

    # Device, confidence, and other options
    device = st.sidebar.selectbox("Device", options=["cpu", "gpu"], index=0)
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold (conf)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
    )
    nms_threshold = st.sidebar.slider(
        "NMS Threshold (nms)", min_value=0.0, max_value=1.0, value=0.3, step=0.01
    )
    test_img_size = st.sidebar.number_input(
        "Test Image Size (tsize)", value=640, step=32
    )

    save_result = st.sidebar.checkbox("Save Inference Result", value=False)
    # Advanced options
    fp16 = st.sidebar.checkbox("Enable FP16 (Mixed Precision)", value=False)
    # legacy = st.sidebar.checkbox("Legacy", value=False)
    trt = st.sidebar.checkbox("Using TensorRT model for testing.", value=False)
    fuse = st.sidebar.checkbox("Fuse Conv and BN for Testing", value=False)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    # Check if a file is uploaded
    if uploaded_file is None:
        # If no file is uploaded, load the default image
        img_name = "car.jpg"
        image = Image.open("assets/car.jpg")
    else:
        image = Image.open(uploaded_file)
        img_name = uploaded_file.name

    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    # Convert to OpenCV BGR format if needed
    image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    args = {
        # "experiment_name": experiment_name,
        "model_name": model_name,
        "path": image_cv2,
        "img_name": img_name,
        "exp_file": exp_file,
        "ckpt": checkpoint,
        "device": device,
        "conf": conf_threshold,
        "nms": nms_threshold,
        "tsize": test_img_size,
        "save_result": save_result,
        "fp16": fp16,
        "legacy": False,
        "fuse": fuse,
        "trt": trt,
    }
    # Display input values
    # st.subheader("Selected Parameters")
    # st.write(args)

    exp = get_exp(exp_file, model_name)

    main(exp, args)
