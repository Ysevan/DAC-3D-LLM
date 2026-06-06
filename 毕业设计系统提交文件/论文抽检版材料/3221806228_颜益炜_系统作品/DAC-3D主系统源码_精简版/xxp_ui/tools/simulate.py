import queue
import random
import threading
import time
import json
import os
from threading import Thread

filename = '../msg/shared_data_win_to_ZYC.txt'
filename_send = '../msg/shared_data_ZYC_to_win.txt'

def read_from_file(filename):
    try:
        with open(filename, "r") as file:
            data = file.read()
        return data, None
    except IOError as e:
        return None, f"Cannot open file: {e}"

def write_to_file(filename, data):
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        with open(filename, "w") as file:
            file.write(data)
    else:
        print(f"File '{filename}' already exists and is not empty.")

def table_to_json(t):
    if isinstance(t, dict):
        return json.dumps(t)
    elif isinstance(t, str):
        return f'"{t}"'
    else:
        return str(t)

def reload_file():
    with open(filename, "w") as file:
        pass

def run_btn():
    print('runBtn clicked')

def stop_btn():
    print('stop clicked')

def receive_data():
    data, err = read_from_file(filename)
    obj = {}
    if data:
        for key, value in [pair.split(':') for pair in data.strip().split(',')]:
            value = value.strip()
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            obj[key] = value
        return obj
    else:
        return None




data = {
    'type': 'new_sample',
    'data': {
        'tray_id': 1,
        'pos': 1,
        'quality': 1,
        'pic_path': 'aaaaa',
        'defection': [
            {
                'defection_type': 'splash',
                'defection_pos': [123, 456],
                'defection_size': 12345
            },
            {
                'defection_type': 'scratch',
                'defection_pos': [741, 852],
                'defection_size': 56789
            },
            {
                'defection_type': 'chipping',
                'defection_pos': [654, 987],
                'defection_size': 75395
            }
        ]
    }
}

def simulate(q,msg):
    print('callback')
    for count in range(1, 145):
        try:
            item = q.get(timeout=0.05)  # 尝试获取队列中的项，超时时间为1秒
            if item == 'STOP':
                print('stop')# 如果收到特殊信号则退出
                break
        except queue.Empty:
            data['data']['tray_id'] = msg.get('Trayid', 1)
            data['data']['pos'] = count
            data['data']['quality'] = 0 if random.random() < 0.5 else 1
            if data['data']['quality'] != 0:
                data['data']['defection'] = []
            else:
                data['data']['defection'] = [
                    {
                        'defection_type': 'splash',
                        'defection_pos': [123, 456],
                        'defection_size': 12345
                    },
                    {
                        'defection_type': 'scratch',
                        'defection_pos': [741, 852],
                        'defection_size': 56789
                    },
                    {
                        'defection_type': 'chipping',
                        'defection_pos': [654, 987],
                        'defection_size': 75395
                    }
                ]

            # print(data['data'])
            json_string = table_to_json(data)
            write_to_file(filename_send, json_string)
            time.sleep(0.1)

while True:
    q = queue.Queue()

    msg = receive_data()
    if msg:
        print(msg)
    # if msg and msg.get('func') == 'run':
    #     print(msg)
    #     run_btn()
    #     worker_thread = threading.Thread(target=simulate, args=(q,msg))
    #     worker_thread.start()
    # elif msg and msg.get('func') == 'stop':
    #     stop_btn()
    #     q.put('STOP')
    reload_file()

