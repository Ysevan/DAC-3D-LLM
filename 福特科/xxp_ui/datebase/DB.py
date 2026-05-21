import datetime
import json
import threading

import pymysql


class DBController:
    lock = threading.Lock()
    history_show_num = 4
    history_begin_num = 0

    def __init__(self):
        self.offline = False
        try:
            self.db = pymysql.connect(
                host='localhost',  # 主机名
                port=3306,  # 端口号，MySQL默认为3306
                user='root',  # 用户名
                password='123456',  # 密码
                database='xxp',  # 数据库名称
            )
            self.cursor = self.db.cursor(pymysql.cursors.DictCursor)
            print('Connected to database')
        except Exception as e:
            self.offline = True
            self.db = None
            self.cursor = None
            self._tray_id = 1
            self._sample_id = 1
            print(f'数据库连接失败，启用离线数据库模式：{e}')

    #
    def databaseSearchAll(self):
        if self.offline:
            return []
        sql = 'select * from sample'
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        self.lock.release()
        print(results)
        return results
        # self.cursor.execute(sql)
        # self.db.commit()

    #
    def addNewTrayToDB(self):
        if self.offline:
            print('[offline-db] 新增工装盘记录')
            self._tray_id += 1
            return
        print('数据库新增工装盘记录')
        sql = "INSERT INTO tray (Detect_time) VALUES(now())"

        self.lock.acquire()
        self.cursor.execute(sql)
        self.db.commit()
        self.lock.release()


    def getLastTrayId(self):
        if self.offline:
            return self._tray_id
        sql = 'SELECT MAX(Tray_id) AS id FROM tray;'
        self.lock.acquire()
        self.cursor.execute(sql)

        self.db.commit()
        results = self.cursor.fetchall()[0]['id']
        self.lock.release()
        if results is None:
            return 1
        return results + 1

    def getLastSampleId(self):
        if self.offline:
            return self._sample_id
        sql = 'SELECT MAX(Sample_id) AS id FROM sample;'
        self.lock.acquire()
        self.cursor.execute(sql)

        self.db.commit()
        results = self.cursor.fetchall()[0]['id']
        self.lock.release()
        if results is None:
            return 1
        return results + 1

    def addNewSampleToDB(self, data):
        if self.offline:
            print(f'[offline-db] 新增样品记录：{data}')
            self._sample_id += 1
            return
        print('数据库新增样品记录')
        Tray_id = data['Tray_id']
        quality = data['quality']
        pic_path = data['pic_path']
        pos = data['pos']
        defection_num = data['defection_num']
        val = (Tray_id, quality, pic_path, pos, defection_num)
        sql = "INSERT INTO sample (Tray_id,quality,pic_path,pos,defection_num) VALUES(%s,%s,%s,%s,%s)"

        self.lock.acquire()
        self.cursor.execute(sql, val)

        self.db.commit()
        # last_insert_id = self.cursor.lastrowid
        # # print(last_insert_id)
        self.lock.release()
        # return last_insert_id

    def addNewDefectionToDB(self, data):
        if self.offline:
            print(f'[offline-db] 新增瑕疵记录：{data}')
            return
        print('数据库新增瑕疵记录')
        Sample_id = data['Sample_id']
        defection_type = data['defection_type']
        defection_pos = data['defection_pos']
        defection_size = data['defection_size']
        val = (Sample_id, defection_type, defection_pos, defection_size)
        sql = "INSERT INTO defection (Sample_id,defection_type,defection_pos,defection_size) VALUES(%s,%s,%s,%s)"

        self.lock.acquire()
        self.cursor.execute(sql, val)

        self.db.commit()
        self.lock.release()

    # def addNewPictureToDB(self, data):
    #     qrcode = data['qrcode']
    #     idtray = data['idtray']
    #     camera = data['camera']
    #     path = data['path']
    #     time = data['time']
    #     picType = data['picType']
    #     val = (idtray, qrcode, path, time, camera, picType)
    #     sql = "INSERT INTO picture (idtray,qrcode,path,time,camera,picType) VALUES(%s,%s,%s,%s,%s,%s)"
    #
    #     self.lock.acquire()
    #     self.cursor.execute(sql, val)
    #     self.db.commit()
    #     self.lock.release()
    #
    # def getDefectionInfo(self, idtray):
    #     sql = 'select * from defections where idtray = %s'
    #     self.lock.acquire()
    #     self.cursor.execute(sql, [idtray])
    #     results = self.cursor.fetchall()
    #     self.lock.release()
    #     # print(results)
    #     return results
    #     # print(results)
    #
    def getTrayInfo(self):
        if self.offline:
            begin = self.history_begin_num
            show = self.history_show_num
            return [], begin, show
        sql = 'select * from tray order by Tray_id desc limit %d,%d' % (self.history_begin_num, self.history_show_num)
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        self.lock.release()
        begin = self.history_begin_num
        show = self.history_show_num
        self.history_begin_num = self.history_begin_num + self.history_show_num
        # print(results)
        return results, begin, show
        # print(results)
    def getOneTray(self,Tray_id):
        if self.offline:
            return {'Tray_id': Tray_id, 'Detect_time': datetime.datetime.now()}
        sql = 'select * from tray where Tray_id=%d' % Tray_id
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()[0]
        self.lock.release()
        return results

    def getSampleBelongToTray(self, Tray_id):
        if self.offline:
            return []
        sql = 'select * from sample where Tray_id=%d' % Tray_id
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        self.lock.release()
        print(results)
        return results

    def getDefectionBelongToSample(self, Sample_id):
        if self.offline:
            return []
        sql = 'select * from defection where Sample_id=%d' % Sample_id
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        self.lock.release()
        print(results)
        return results

    def getTrayDetectTime(self, Tray_id):
        if self.offline:
            return str(datetime.datetime.now()).split('.')[0]
        sql = 'select * from tray where Tray_id=%d' % Tray_id
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        self.lock.release()
        print(results)
        return str(results[0]['Detect_time'])

    # 推送
    def getDefectionByPos(self, Tray_id, pos):
        if self.offline:
            return []
        sql = 'select * from sample where Tray_id=%d and pos=%d' % (Tray_id, pos)
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()[0]['Sample_id']
        self.lock.release()
        sql = 'select * from defection where Sample_id=%d' % results
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        self.lock.release()
        print(results)
        return results

    def countSampleOfTray(self,Tray_id):
        if self.offline:
            return 0, 0
        sql = f'SELECT SUM(CASE WHEN quality = 1 THEN 1 ELSE 0 END) AS 合格_count,SUM(CASE WHEN quality != 1 THEN 1 ELSE 0 END)  AS 不合格_count FROM sample WHERE Tray_id = {Tray_id};'
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()[0]
        self.lock.release()
        return int(results['合格_count']),int(results['不合格_count'])

    def countDefectionOfSample(self, Tray_id):
        if self.offline:
            return []
        sql = f'''
            SELECT 
                d.defection_type, 
                COUNT(d.Defection_id) AS defect_count
            FROM 
                tray t
            JOIN 
                sample s ON t.Tray_id = s.Tray_id
            JOIN 
                defection d ON s.Sample_id = d.Sample_id
            WHERE 
                t.Tray_id = {Tray_id}  -- 替换为你要查询的具体 Tray_id
            GROUP BY 
                d.defection_type;
        '''
        self.lock.acquire()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        self.lock.release()
        # print(results)
        return results
    #
    # # def getNewIdTray(self):
    # #     # sql = 'select idtray from tray'
    # #     sql = 'SELECT @@IDENTITY'
    # #     self.lock.acquire()
    # #     self.cursor.execute(sql)
    # #     results = self.cursor.fetchall()
    # #     self.lock.release()
    # #     print(results)
    # #     # return results[-1]['idtray']
    def clearDB(self):
        if self.offline:
            print('[offline-db] clearDB')
            return
        sql1 = 'TRUNCATE TABLE tray;'
        sql2 = 'ALTER TABLE tray AUTO_INCREMENT = 1;'

        sql3 = 'TRUNCATE TABLE sample;'
        sql4 = 'ALTER TABLE sample AUTO_INCREMENT = 1;'

        sql5 = 'TRUNCATE TABLE defection;'
        sql6 = 'ALTER TABLE defection AUTO_INCREMENT = 1;'

        self.lock.acquire()
        self.cursor.execute(sql1)
        self.lock.release()
        self.lock.acquire()
        self.cursor.execute(sql2)
        self.lock.release()
        self.lock.acquire()
        self.cursor.execute(sql3)
        self.lock.release()
        self.lock.acquire()
        self.cursor.execute(sql4)
        self.lock.release()
        self.lock.acquire()
        self.cursor.execute(sql5)
        self.lock.release()
        self.lock.acquire()
        self.cursor.execute(sql6)
        self.lock.release()
        return


if __name__ == '__main__':
    myDatabase = DBController()
    # myDatabase.addNewSampleToDB()
    # myDatabase.getSampleBelongToTray(7)
    # print(myDatabase.getOneTray(7))
    # myDatabase.history_begin_num = 0

    # myDatabase.getDefectionByPos(62,2)
    # historyInfo, begin, show_num = myDatabase.getTrayInfo()
    # print(historyInfo, begin, show_num)
    # data = {'Tray_id': 1, 'quality': 1, 'pic_path': 'aaa', 'pos': 38, 'defection_num': 5}
    # myDatabase.addNewSampleToDB(data)
    # res = myDatabase.databaseSearchAll()
    # test_data = {
    #     "qrcode": 123456,
    #     "diameter": 2,
    #     'totalNum': 3,
    #     'creationtime': '2024-3-18 16:22:18',
    # }
    #
    # data = {}
    # data['Sample_id'] = 1
    # data['defection_type'] = 'splash'
    # data['defection_pos'] ="1,2"
    # data['defection_size']=35.5

    # print(myDatabase.getLastSampleId())
    # test_data_2 = {
    #     "qrcode": 'sdadsadsaf',
    #     'path': 'sdadsdasda',
    #     'time': '2024-3-19 10:45:51',
    #     'camera': 1,
    #     'picType': 'origin'
    # }

    # myDatabase.databaseSearchAll()

    # myDatabase.addNewTrayToDB(test_data)
    # myDatabase.addNewDefectionToDB(test_data_1)
    # myDatabase.addNewPictureToDB(test_data_2)
    # sql = 'select * from tray'
    # myDatabase.databaseControl(sql)

    # myDatabase.getDefectionInfo('sdhosjnxualsnxksiwtdbsjakl')
    # myDatabase.getNewIdTray()
    # myDatabase.getDefectionInfo(25)
    # print(myDatabase.getTrayInfo())
    # print(myDatabase.getTrayInfo())

    print(type(str((1,2))))
