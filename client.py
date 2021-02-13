import aiohttp
import asyncio
import os


IMGS_FOLDER = "media"


async def classification_preprocess(session, url, data):
    async with session.post(url, data=data) as response:
        if response.method == 'POST':
            content_type = response.content_type
            if content_type == 'multipart/form-data':
                reader = aiohttp.MultipartReader.from_response(response)

                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    else:
                        content_type_part = part.headers
                        if content_type_part['Content-Type'] == 'application/json':
                            data_json = await part.json()
                            return data_json['data']['img_classes']


async def main():
    async with aiohttp.ClientSession() as session:
        list_imgs = os.listdir(IMGS_FOLDER)

        with aiohttp.MultipartWriter('form-data') as mpwriter:
            with aiohttp.MultipartWriter('related') as subwriter:
                for img_file_name in list_imgs:
                    full_file_name = os.path.join(IMGS_FOLDER, img_file_name)
                    img_bytes = open(full_file_name, "rb").read()
                    subwriter.append(img_bytes)
            mpwriter.append(subwriter)

            mpwriter.append_json({
                "process": "image_classification",
                "list_files": list_imgs
            })

            classes = await classification_preprocess(session, 'http://localhost:8080/predict', data=mpwriter)
            print("ML CLASSIFICATION\nResult:")
            for img in classes:
                print("File: {0}, class: {1}".format(img, classes[img]))


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
