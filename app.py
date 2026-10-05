
import base64
import io
import random
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# =========================================================
# 1. 项目路径
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RESNET_PATH = BASE_DIR / "models" / "moodmate_finetuned.pth"
YUNET_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"


# =========================================================
# 2. 加载 ResNet18 表情识别模型
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

emotion_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    7
)

model.load_state_dict(
    torch.load(
        RESNET_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================================================
# 3. 图片预处理
# =========================================================

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 4. 加载 YuNet 人脸检测器
# =========================================================

detector = cv2.FaceDetectorYN.create(
    str(YUNET_PATH),
    "",
    (320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)


# =========================================================
# 5. ResNet18 单张人脸预测
# =========================================================

def predict_emotion(image):

    image = image.convert("RGB")

    image_tensor = (
        eval_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    emotion = emotion_names[
        predicted.item()
    ]

    return emotion, confidence.item()


# =========================================================
# 6. YuNet 裁剪人脸
# =========================================================

def extract_face_from_pil(pil_image):

    if pil_image is None:
        return None

    image_rgb = pil_image.convert("RGB")

    image_cv = cv2.cvtColor(
        np.array(image_rgb),
        cv2.COLOR_RGB2BGR
    )

    h, w = image_cv.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(image_cv)

    if faces is None or len(faces) == 0:
        return None

    # 如果同时出现多张脸，只处理最大的人脸
    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    x, y, fw, fh = face[:4].astype(int)

    # 给检测框多留一点空间
    pad_x = int(fw * 0.25)
    pad_y = int(fh * 0.25)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)

    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)

    return image_rgb.crop(
        (x1, y1, x2, y2)
    )


# =========================================================
# 7. 情绪 → 页面颜色
# =========================================================

emotion_color_map = {
    "angry": "#FF5D5D",
    "disgust": "#8ED35F",
    "fear": "#9B82E3",
    "happy": "#FFD400",
    "sad": "#5D91F2",
    "surprise": "#FF9D35",
    "neutral": "#E8DDC8",
}


emotion_cn_map = {
    "angry": "生气",
    "disgust": "嫌弃",
    "fear": "害怕",
    "happy": "开心",
    "sad": "难过",
    "surprise": "震惊",
    "neutral": "平静",
}


# =========================================================
# 8. 阴阳怪气文案
# 每次只随机显示一句
# =========================================================

emotion_lines = {

    "angry": [
        "这就破防？小妹妹你蛮牛逼克拉斯啊。",
        "哟，谁又惹到你了，火气这么冲。",
        "谁把你 CPU 干烧了？",
        "这个脸色有点东西，下一秒准备开喷是吧。",
        "表情管理已经正式宣布下班。",
        "谁又这么精准地踩到你的雷区了？",
        "今天进入战斗模式了是吧。",
        "别急，互联网还不值得你这么认真。"
    ],

    "disgust": [
        "这脸色，一看就是：晦气，退退退。",
        "你这不是无语，你这是从心底嫌弃。",
        "这个表情已经替你打完一星差评了。",
        "谁又整了个离谱操作把你恶心到了？",
        "懂了，这个东西不配进入您的法眼。",
        "白眼虽然没翻，但精神上已经翻了。",
        "这个嫌弃多少有点溢出屏幕了。",
        "建议直接划走，别污染心情。"
    ],

    "fear": [
        "怎么，老板突然 @ 你了？",
        "这个表情像是突然想起今天有截止日期。",
        "不是，谁吓你了？",
        "你现在这个状态像 CPU 在疯狂弹窗。",
        "这个眼神有种论文马上送审的美。",
        "慌什么，天暂时还没塌。",
        "你这脸上已经写满了：完了完了完了。",
        "先别自己吓自己，问题可能还没加载出来。"
    ],

    "happy": [
        "666，拿到大结果了不告诉我？",
        "怎么偷偷开心上了，说吧，谁给你发好消息了。",
        "哟，嘴角压不住了，还装呢。",
        "看你这样子，今天多少有点顺。",
        "好好好，背着我发财是吧。",
        "这个笑容多少有点故事。",
        "笑这么明显，有喜事不准备分我一点？",
        "今天状态直接起飞了是吧。"
    ],

    "sad": [
        "哟，又网易云上了。",
        "怎么突然进入深夜电台模式了？",
        "谁把你的情绪 KPI 干没了？",
        "这个脸看起来多少被生活教育了一下。",
        "好家伙，直接 emo 启动。",
        "别低头，皇冠会掉，虽然你也没皇冠。",
        "今天先这样，明天再重新做人。",
        "这个表情主打一个：算了。"
    ],

    "surprise": [
        "哟，这是看到什么大瓜了？",
        "眼睛睁这么大，Excel 又炸了？",
        "不是，发生什么了？",
        "怎么突然震惊部入驻了。",
        "这个反应多少有点节目效果。",
        "好家伙，CPU 突然拉满。",
        "世界一直这么癫，习惯就好。",
        "你这个表情已经自带三个感叹号了。"
    ],

    "neutral": [
        "CPU 在线，情绪模块暂未启动。",
        "今天是电子木鱼模式是吧。",
        "你现在这个表情像个默认头像。",
        "好一个无悲无喜。",
        "今天属于低功耗运行。",
        "灵魂在线，但好像没完全上线。",
        "这个状态很适合装高手。",
        "主打一个：哦，然后呢。"
    ]
}


# =========================================================
# 9. 页面 HTML
# =========================================================

def build_page_style(color):

    return f"""
    <style>

        body {{
            background: {color} !important;
        }}

        .gradio-container {{
            background: {color} !important;
        }}

    </style>
    """


def build_waiting_panel():

    return """
    <div class="emotion-panel">

        <div class="emotion-small">
            MOODMATE SEES
        </div>

        <div class="emotion-title">
            READY
        </div>

        <div class="emotion-cn">
            等你表演
        </div>

        <div class="message-box">
            看镜头，然后点一下识别。
        </div>

    </div>
    """


def build_no_face_panel():

    return """
    <div class="emotion-panel">

        <div class="emotion-small">
            MOODMATE SEES
        </div>

        <div class="emotion-title">
            OOPS
        </div>

        <div class="emotion-cn">
            没找到脸
        </div>

        <div class="message-box">
            人呢？镜子看了个寂寞。
        </div>

    </div>
    """


def build_emotion_panel(
    emotion,
    confidence,
    one_line
):

    emotion_cn = emotion_cn_map[
        emotion
    ]

    return f"""
    <div class="emotion-panel">

        <div class="emotion-small">
            MOODMATE SEES
        </div>

        <div class="emotion-title">
            {emotion.upper()}
        </div>

        <div class="emotion-cn">
            {emotion_cn}
        </div>

        <div class="confidence-line">
            CONFIDENCE · {confidence:.0%}
        </div>

        <div class="message-box">
            {one_line}
        </div>

    </div>
    """


# =========================================================
# 10. 接收浏览器截图并识别
# =========================================================

def analyze_snapshot(snapshot_data):

    if not snapshot_data:

        return (
            build_waiting_panel(),
            build_page_style("#E8DDC8")
        )

    try:

        if "," in snapshot_data:

            snapshot_data = (
                snapshot_data
                .split(",", 1)[1]
            )

        image_bytes = base64.b64decode(
            snapshot_data
        )

        frame = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        return (
            build_no_face_panel(),
            build_page_style("#D8D8D8")
        )

    face = extract_face_from_pil(
        frame
    )

    if face is None:

        return (
            build_no_face_panel(),
            build_page_style("#D8D8D8")
        )

    emotion, confidence = predict_emotion(
        face
    )

    one_line = random.choice(
        emotion_lines[emotion]
    )

    return (
        build_emotion_panel(
            emotion,
            confidence,
            one_line
        ),

        build_page_style(
            emotion_color_map[emotion]
        )
    )


# =========================================================
# 11. CSS
# =========================================================

custom_css = """

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    min-height: 100vh !important;
    transition: background 0.35s ease !important;
}


/* 标题 */

#main-title {
    text-align: center;
    padding-top: 28px;
    margin-bottom: 30px;
}

#main-title h1 {
    margin: 0;
    font-size: 58px;
    font-weight: 900;
    letter-spacing: -3px;
    color: #202020;
}


/* ==========================
   完整魔镜
========================== */

#mirror-wrapper {

    width: 100%;

    padding: 18px;

    box-sizing: border-box;

    border-radius: 180px;

    background:
        linear-gradient(
            135deg,
            #805000 0%,
            #D5960C 14%,
            #FFE995 26%,
            #9B6300 40%,
            #FFD55B 55%,
            #805000 70%,
            #F9D967 84%,
            #845000 100%
        );

    box-shadow:
        0 18px 45px
        rgba(0,0,0,0.20);
}


#mirror-inner {

    position: relative;

    overflow: hidden;

    border-radius: 165px;

    border:
        8px solid
        #FFF2C0;

    background: #111;
}


/* 实时video */

#mood-video {

    display: block;

    width: 100%;

    aspect-ratio: 16 / 9;

    object-fit: cover;

    transform: scaleX(-1);

    background: #111;
}


/* 0.5秒冻结画面 */

#freeze-canvas {

    display: none;

    position: absolute;

    inset: 0;

    z-index: 5;

    width: 100%;

    height: 100%;
}


/* 快门白闪 */

#flash-layer {

    position: absolute;

    inset: 0;

    z-index: 10;

    pointer-events: none;

    background: white;

    opacity: 0;

    transition: opacity 0.1s;
}


/* 按钮 */

#capture-button button {

    width: 100%;

    margin-top: 20px;

    border-radius: 20px !important;

    padding: 15px !important;

    border: none !important;

    background: #181818 !important;

    color: white !important;

    font-size: 19px !important;

    font-weight: 900 !important;
}


/* 右边 */

.emotion-panel {

    min-height: 430px;

    padding: 35px 20px;

    color: #202020;
}


.emotion-small {

    margin-bottom: 12px;

    font-size: 15px;

    font-weight: 800;

    letter-spacing: 6px;

    opacity: 0.65;
}


.emotion-title {

    margin-bottom: 18px;

    font-size: 96px;

    line-height: 0.92;

    font-weight: 900;

    letter-spacing: -5px;
}


.emotion-cn {

    margin-bottom: 18px;

    font-size: 35px;

    font-weight: 900;
}


.confidence-line {

    margin-bottom: 38px;

    font-size: 17px;

    font-weight: 800;

    letter-spacing: 2px;

    opacity: 0.65;
}


.message-box {

    max-width: 95%;

    font-size: 38px;

    line-height: 1.4;

    font-weight: 900;
}


/* 隐藏截图数据框 */

#snapshot-data {
    display: none !important;
}


footer {
    display: none !important;
}

"""


# =========================================================
# 12. 页面启动时打开摄像头
# =========================================================

custom_js = r"""
async () => {

    await new Promise(
        resolve => setTimeout(resolve, 600)
    );

    const video =
        document.getElementById("mood-video");

    if (!video) {
        return;
    }

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({

                video: {

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    },

                    facingMode: "user"
                },

                audio: false
            });

        video.srcObject = stream;

        await video.play();

    }

    catch (error) {

        console.error(
            "Camera error:",
            error
        );
    }
}
"""


# =========================================================
# 13. Gradio 页面
# =========================================================

with gr.Blocks() as demo:

    page_style_output = gr.HTML(
        build_page_style("#E8DDC8")
    )

    gr.HTML(
        """
        <div id="main-title">
            <h1>MOODMATE</h1>
        </div>
        """
    )

    with gr.Row():

        # 左边魔镜
        with gr.Column(scale=6):

            gr.HTML(
                """
                <div id="mirror-wrapper">

                    <div id="mirror-inner">

                        <video
                            id="mood-video"
                            autoplay
                            muted
                            playsinline>
                        </video>

                        <canvas
                            id="freeze-canvas">
                        </canvas>

                        <div
                            id="flash-layer">
                        </div>

                    </div>

                </div>
                """
            )

            snapshot_data = gr.Textbox(
                value="",
                visible=True,
                elem_id="snapshot-data"
            )

            capture_btn = gr.Button(
                "识别一下",
                elem_id="capture-button"
            )

        # 右边结果
        with gr.Column(scale=6):

            emotion_panel = gr.HTML(
                build_waiting_panel()
            )


    # =========================================================
    # 14. 点击“识别一下”
    # =========================================================

    capture_btn.click(

        fn=analyze_snapshot,

        inputs=snapshot_data,

        outputs=[
            emotion_panel,
            page_style_output
        ],

        js=r"""
        async (old_value) => {

            const video =
                document.getElementById(
                    "mood-video"
                );

            const freezeCanvas =
                document.getElementById(
                    "freeze-canvas"
                );

            const flash =
                document.getElementById(
                    "flash-layer"
                );

            if (
                !video ||
                video.videoWidth === 0
            ) {
                return "";
            }

            // -------------------------
            // 截当前视频帧
            // -------------------------

            const shot =
                document.createElement(
                    "canvas"
                );

            shot.width =
                video.videoWidth;

            shot.height =
                video.videoHeight;

            const ctx =
                shot.getContext("2d");

            // 镜像截图
            ctx.translate(
                shot.width,
                0
            );

            ctx.scale(
                -1,
                1
            );

            ctx.drawImage(
                video,
                0,
                0,
                shot.width,
                shot.height
            );

            // -------------------------
            // 0.5秒冻结效果
            // -------------------------

            freezeCanvas.width =
                shot.width;

            freezeCanvas.height =
                shot.height;

            const freezeCtx =
                freezeCanvas.getContext(
                    "2d"
                );

            freezeCtx.drawImage(
                shot,
                0,
                0
            );

            freezeCanvas.style.display =
                "block";

            // 白色快门闪一下
            flash.style.opacity = "0.7";

            setTimeout(
                () => {
                    flash.style.opacity = "0";
                },
                100
            );

            // 0.5秒恢复实时画面
            setTimeout(
                () => {
                    freezeCanvas.style.display =
                        "none";
                },
                500
            );

            // -------------------------
            // 图片传给Python
            // -------------------------

            return shot.toDataURL(
                "image/jpeg",
                0.92
            );
        }
        """
    )



# =========================================================
# 15. 启动
# =========================================================

if __name__ == "__main__":

    demo.launch(
        share=True,
        css=custom_css,
        js=custom_js
    )
