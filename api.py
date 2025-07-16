import time

import pandas as pd
import requests
import asyncio
import aiohttp
resumes = [
        """个人简历 细心从每一个小细节开始。
Personal resume
基本信息
姓 名 :赖科熊 出生年月:1994.08.06
民 族 :汉 身 高:171cm
电 话 :18889255699 毕业院校 :兰州理工大学
邮 箱:1173336468 @qq.com 学 历 :本科
住 址:海南省万宁市大茂镇厚 求职意向:电气工程师
皮树村十七队
教育背景
2016.09-2021.06 兰州理工大学 电气工程及其自动化 本科
主修课程:模拟电子技术、电机学、电力电子技术、自动控制原理、数字电子技术、运动控制系统,电气检测技
术、计算机控制技术等
辅修课程:电气制图与 CAD、特种电机及其应用、电气工程新技术专题等
校园经历
2017.10 秋季加入羽毛球协会,增加各个方面的能力
2018.07 在广意广告公司担任设计师,主要负责客户的广告设计,丰富了许多知识。
2019.06 参加电子工艺实习,提升动手操作能力
2019.06 参加认知实习,成绩良好
技能证书
2019.05 获得中级电工证
自我评价
在校期间,各个方面表现良好。在生活方面,积极参加各项活动,做事耐心勤快,认真负责,与同学关系融洽,
由良好的人际关系。在学习方面,勤奋好学,专业课成绩良好,选修电子电路制作实践课程,具有良好的动手操作
能力,自学 CAD 制图,并取得相关证书,具备较强的自学能力。""",

        """邱翔
电话: 19907956665 |邮箱: 970061872@qq.com
现居城市: 江西省宜春市袁州区朝阳路小区
年龄: 28岁 |性别: 男 |身高: 182 |体重: 80 |籍贯: 江西省 |民族: 汉族 |政治面貌: 共青团员 |婚姻状况: 未婚
当前状态: 在职,湖南长沙 |求职意向: 猪场设计规划或相关养殖业务工作
工作经历
湖南神诸科技有限公司 2020年03月
设计规划 技术部 长沙
主要服务甲方的猪场规划设计,从前期的现场看地,再到拿到红线图后根据甲方规模确定工艺和总平图纸,之后再配合结构,
给排水,电气等深化图纸,交付给甲方进行施工,并配合解决施工当中的问题。同时还负责公司鸟瞰图制作,效果表达
主导设计项目:
(1)涟源600头母猪自繁自养
(2)湖南宝泽农牧科技有限公司2400头母猪场
(3)衡阳宇昇农牧万头育肥厂
(4)花垣裕隆团结300母猪厂
(5)湘潭闽企2000母猪厂
(6)湘潭闽企20000育肥厂
(7)中湖农牧发展有限公司2200母猪场
教育经历
江西工业职业技术学院
南昌
室内装潢设计 大专
其他
技能: 装饰美工中级
证书/执照: 装饰美工中级
个人总结
熟悉现代化猪场设备和应用
对猪舍的冬季夏季的各种通风模式优缺点熟悉
熟练实用设计专业软件和鸟瞰图制作
沟通能力强,具有团队协作精神,有较高的执行力,合作共赢
有问题会钻研,并且总结
"""
    ]*10
"""
chunk_size=20
for chunk in pd.read_csv("./db_ai.t_ck_cand_con0204_trunk.csv/db_ai.t_ck_cand_con0204_trunk.csv",
                             chunksize=chunk_size):
    resumes=list(chunk.iloc[:,2])
    break
resumes_dict= [{str(i):j for i,j in enumerate(resumes)}]
"""

from pymongo import MongoClient
client = MongoClient('mongodb://resume_db_test:F1rOLEw3OyRu558mROeX@dds-bp1f9e511ec921841.mongodb.rds.aliyuncs.com:3717/resume_db_test?authMechanism=SCRAM-SHA-1')
print(client.list_database_names())
db = client['resume_db_test']
collection = db['resumeCollection']
query = {
    "status": {"$in": [6]},
    "corpCode": "3DMed"
}

results = collection.find(query)
resumes={}
resumes_dict=[]
i=0
for result in results:
    # resumeId,resumeContent
    i+=1
    if i>20:
        break
    resumes[result['resumeId']]=result['resumeContent']


resumes_dict.append(resumes)
#print(resumes_dict)

async def send_request(session, resume):
    async with session.post('http://172.16.2.21:8009/summary', headers={'token':'d21b5ecb0ade56ca789a959e2bb57074'},json={"text": resume}) as response:
        print(f"Status: {response.status}")
        data = await response.json()
        print("Response:", data)
        return data

async def main():
    connector = aiohttp.TCPConnector(limit_per_host=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request(session, resume) for resume in resumes_dict]
        results = await asyncio.gather(*tasks)
        return results

if __name__ == '__main__':
    collection1 = db['resume_summary']
    start=time.time()
    result=asyncio.run(main())
    end=time.time()
    print(end-start)
    for key,value in result[0].items():
        many_data = [{'resumeId':int(key),'resume':resumes[int(key)],'resume_summary':value}]
        collection1.insert_many(many_data)

