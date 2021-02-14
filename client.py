import aiohttp
import asyncio
import os
import time
from codetiming import Timer


IMGS_FOLDER = "media"


async def task(name, work_queue):
    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
    while not work_queue.empty():
        delay = await work_queue.get()
        print(f"Task {name} running")
        timer.start()

        async with aiohttp.ClientSession() as session:
            classes = await classification_preprocess(session,
                                                      'http://localhost:8080/predict',
                                                      data=delay)

        timer.stop()
        return classes


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
    list_imgs = os.listdir(IMGS_FOLDER)

    # Создание 1 очереди работы
    work_queue1 = asyncio.Queue()

    # Помещение работы в очередь
    with aiohttp.MultipartWriter('form-data') as mpwriter:
        with aiohttp.MultipartWriter('related') as subwriter:
            full_file_name = os.path.join(IMGS_FOLDER, list_imgs[0])
            img_bytes = open(full_file_name, "rb").read()
            subwriter.append(img_bytes)
        mpwriter.append(subwriter)

        mpwriter.append_json({
            "process": "image_classification",
            "list_files": [list_imgs[0]]
        })
    await work_queue1.put(mpwriter)

    # Создание 2 очереди работы
    work_queue2 = asyncio.Queue()

    # Помещение работы в очередь
    with aiohttp.MultipartWriter('form-data') as mpwriter:
        with aiohttp.MultipartWriter('related') as subwriter:
            full_file_name = os.path.join(IMGS_FOLDER, list_imgs[1])
            img_bytes = open(full_file_name, "rb").read()
            subwriter.append(img_bytes)
        mpwriter.append(subwriter)

        mpwriter.append_json({
            "process": "image_classification",
            "list_files": [list_imgs[1]]
        })
    await work_queue2.put(mpwriter)

    print(f"started at {time.strftime('%X')}")

    # Запуск задач
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        results = await asyncio.gather(
            asyncio.create_task(task("One", work_queue1)),
            asyncio.create_task(task("Two", work_queue2)),
        )
        print('='*10 + "\nML CLASSIFICATION\nResult:")

        for result in results:
            for file in result:
                print("File: {0}, class: {1}".format(file, result[file]))

    print(f"finished at {time.strftime('%X')}")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
