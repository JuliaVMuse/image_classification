from aiohttp import web, MultipartWriter
from PIL import Image
from io import BytesIO
import numpy as np


def classification(img):
    """
    Predict class
    """

    return "flower"


def ml_classification(images, name_files):
    """
    ML-classification images
    """

    classes = dict()

    for name_file, image in zip(name_files, images):
        stream = BytesIO(image)
        orig_img = Image.open(stream).convert("RGB")
        stream.close()
        img_class = classification(orig_img)
        classes[name_file] = img_class

    return classes


def get_image_classes(images, files):
    print("get_image_classes")
    img_classes = ml_classification(images, files)

    with MultipartWriter('form-data') as mpwriter:
        mpwriter.append_json({
            "data": {"img_classes": img_classes}
        })

    return mpwriter


async def predict(request):
    if request.method == 'POST':
        imgs_bytes = list()
        process, name_files = None, None
        content_type = request.content_type
        if content_type == 'multipart/form-data':
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                else:
                    content_type_part = part.headers
                    if content_type_part['Content-Type'] == 'application/json':
                        data_json = await part.json()
                        process = data_json['process']
                        name_files = data_json['list_files']
                    else:
                        while True:
                            sub_part = await part.next()
                            if sub_part is None:
                                break
                            else:
                                while True:
                                    sub_part1 = await sub_part.next()
                                    if sub_part1 is None:
                                        break
                                    else:
                                        imgs_bytes.append(bytes(sub_part1))
                                        continue

        if process == 'image_classification':
            mpwriter = get_image_classes(imgs_bytes, name_files)
            return web.Response(body=mpwriter)


app = web.Application()
app.add_routes([web.post('/predict', predict)])


if __name__ == '__main__':
    web.run_app(app, host='localhost', port=8080)
