import requests
import sys
import io
import json
import csv
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
f = open('job_data_Supplychain.csv', mode='w', encoding='utf-8-sig', newline='')
csv_writer = csv.DictWriter(f, fieldnames=[
    '职位名称', '公司名称', '薪资', '工作城市', '工作城区',
    '发布时间', '工作经验要求', '学历要求', '职位标签',
    '职位描述', '公司类型', '公司规模', '职位链接'
])
csv_writer.writeheader()
import requests


cookies = {
    'sajssdk_2015_cross_new_user': '1',
    'guid': '6095be4a0a003bb43285f7eb99a573ff',
    'sensorsdata2015jssdkcross': '%7B%22distinct_id%22%3A%226095be4a0a003bb43285f7eb99a573ff%22%2C%22first_id%22%3A%2219a9b0637aa493-0cb97c620ba8a18-26061b51-1382400-19a9b0637abead%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTlhOWIwNjM3YWE0OTMtMGNiOTdjNjIwYmE4YTE4LTI2MDYxYjUxLTEzODI0MDAtMTlhOWIwNjM3YWJlYWQiLCIkaWRlbnRpdHlfbG9naW5faWQiOiI2MDk1YmU0YTBhMDAzYmI0MzI4NWY3ZWI5OWE1NzNmZiJ9%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%226095be4a0a003bb43285f7eb99a573ff%22%7D%2C%22%24device_id%22%3A%2219a9b0637aa493-0cb97c620ba8a18-26061b51-1382400-19a9b0637abead%22%7D',
    'tfstk': 'gLgrnfj3pULrJvXXPxUF_Eket9z8-yJ6Y2wQtXc3N82lPQnVoWM7FBhHevuUnX07E8O88kDnEvrpwJw3LvGnVIT65bh8JyVeCFT_KDVIpY7uZ8A3m5N6G7Yt4_YLJyv6hychBuzKU37BZ8cDgWNC-wD3qr40N-V3ZyD3noVg1T43-vA4m5NlqMV3xrV0N-23-yDniIPLn743-vcDgWhrDy4O3WZkG9ee59-FUowiZ-7hSsF4qQlRANbiS7rrw-w4a7u4auygYWAKw4Djt4g8DnS0JjiZLDDMhgwob5k4f0JNzAk8tAzSlTA0-kmjfuo2EwyjPWrg-o5h-jzj0ku4SFX4MmcSx4HPTwV-PVZUBoRhJkam5lmiU6dtizVnpl3BBaeotf3tfP8PB7om_rSz99F0fLgK49jUqSF4CIRVVncHwlsBEHjdvuCYgROuqMILqSF4CIRVvMEJHSy6Zuf..',
    'JSESSIONID': 'FD2C5B3243EB0A1D292B5310E00C930B',
    'acw_tc': 'ac11000117635411757274603e009f84ef013ad1f0ab4d386ad6213093f937',
    'acw_sc__v2': '691d80b95ad99ae67183bd80f782ae94b2fe5539',
    'ssxmod_itna': '1-YuiQ0KBIx0ObDQG03GCmaG70q7ItKGHDyxWFK0CDmxjKidqDUnPHDiw87r_5i=9PjDE0G3hbAqDsPSPfeDyCxA3DTPRp=KinEDbNIPF/Abx19jxoOpqEXhto00EkstZSxdLLI7Rgh840aDm9T1QGtQe4DxxGTDCeDQxirDD4DADibNxD1pDDkD04p89EvF4GWDm4p34GCDQIpDiPD=6fw63xD0xD84/33H4xDBxh2imDDecTNLIjG6FRLk9rRxroD9h4Dscu9iiEUCM=e_ICcxp8bt40kjq0OzFRANZ8QHbSvEKRD9mWVh7d=BNdEqPGxaD=1BD9Ye1Yx1A4ZBDLO_hDYQBKq7YQW5xnwkhFsCdkxY5Ye6qzRwz9deof5ZoI/ghqG=Tmxo0YpO5/YtmrdpE5a7=97xZBiPiGaiDD',
    'ssxmod_itna2': '1-YuiQ0KBIx0ObDQG03GCmaG70q7ItKGHDyxWFK0CDmxjKidqDUnPHDiw87r_5i=9PjDE0G3hbe4iThGe_Euv=kFi5XtV4FEDwhrZ4rD',
}

headers = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-US,en;q=0.9',
    'account-id': '',
    'from-domain': '51job_web',
    'partner': '',
    'priority': 'u=1, i',
    'property': '%7B%22partner%22%3A%22%22%2C%22webId%22%3A2%2C%22fromdomain%22%3A%2251job_web%22%2C%22frompageUrl%22%3A%22https%3A%2F%2Fwe.51job.com%2F%22%2C%22pageUrl%22%3A%22https%3A%2F%2Fwe.51job.com%2Fpc%2Fsearch%3Fkeyword%3D%25E4%25BE%259B%25E5%25BA%2594%25E9%2593%25BE%26searchType%3D2%26keywordType%3D%22%2C%22identityType%22%3A%22%22%2C%22userType%22%3A%22%22%2C%22isLogin%22%3A%22%E5%90%A6%22%2C%22accountid%22%3A%22%22%2C%22keywordType%22%3A%22%22%7D',
    'referer': 'https://we.51job.com/pc/search?keyword=%E4%BE%9B%E5%BA%94%E9%93%BE&searchType=2&keywordType=',
    'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    #'sign': '983a433097649bc948f9de5843a576ab9b72a6bb8ae7a438f64634dbc5b0b271',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
    'user-token': '',
    'uuid': '6095be4a0a003bb43285f7eb99a573ff',
    # 'cookie': 'sajssdk_2015_cross_new_user=1; guid=6095be4a0a003bb43285f7eb99a573ff; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%226095be4a0a003bb43285f7eb99a573ff%22%2C%22first_id%22%3A%2219a9b0637aa493-0cb97c620ba8a18-26061b51-1382400-19a9b0637abead%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTlhOWIwNjM3YWE0OTMtMGNiOTdjNjIwYmE4YTE4LTI2MDYxYjUxLTEzODI0MDAtMTlhOWIwNjM3YWJlYWQiLCIkaWRlbnRpdHlfbG9naW5faWQiOiI2MDk1YmU0YTBhMDAzYmI0MzI4NWY3ZWI5OWE1NzNmZiJ9%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%226095be4a0a003bb43285f7eb99a573ff%22%7D%2C%22%24device_id%22%3A%2219a9b0637aa493-0cb97c620ba8a18-26061b51-1382400-19a9b0637abead%22%7D; tfstk=gLgrnfj3pULrJvXXPxUF_Eket9z8-yJ6Y2wQtXc3N82lPQnVoWM7FBhHevuUnX07E8O88kDnEvrpwJw3LvGnVIT65bh8JyVeCFT_KDVIpY7uZ8A3m5N6G7Yt4_YLJyv6hychBuzKU37BZ8cDgWNC-wD3qr40N-V3ZyD3noVg1T43-vA4m5NlqMV3xrV0N-23-yDniIPLn743-vcDgWhrDy4O3WZkG9ee59-FUowiZ-7hSsF4qQlRANbiS7rrw-w4a7u4auygYWAKw4Djt4g8DnS0JjiZLDDMhgwob5k4f0JNzAk8tAzSlTA0-kmjfuo2EwyjPWrg-o5h-jzj0ku4SFX4MmcSx4HPTwV-PVZUBoRhJkam5lmiU6dtizVnpl3BBaeotf3tfP8PB7om_rSz99F0fLgK49jUqSF4CIRVVncHwlsBEHjdvuCYgROuqMILqSF4CIRVvMEJHSy6Zuf..; JSESSIONID=FD2C5B3243EB0A1D292B5310E00C930B; acw_tc=ac11000117635411757274603e009f84ef013ad1f0ab4d386ad6213093f937; acw_sc__v2=691d80b95ad99ae67183bd80f782ae94b2fe5539; ssxmod_itna=1-YuiQ0KBIx0ObDQG03GCmaG70q7ItKGHDyxWFK0CDmxjKidqDUnPHDiw87r_5i=9PjDE0G3hbAqDsPSPfeDyCxA3DTPRp=KinEDbNIPF/Abx19jxoOpqEXhto00EkstZSxdLLI7Rgh840aDm9T1QGtQe4DxxGTDCeDQxirDD4DADibNxD1pDDkD04p89EvF4GWDm4p34GCDQIpDiPD=6fw63xD0xD84/33H4xDBxh2imDDecTNLIjG6FRLk9rRxroD9h4Dscu9iiEUCM=e_ICcxp8bt40kjq0OzFRANZ8QHbSvEKRD9mWVh7d=BNdEqPGxaD=1BD9Ye1Yx1A4ZBDLO_hDYQBKq7YQW5xnwkhFsCdkxY5Ye6qzRwz9deof5ZoI/ghqG=Tmxo0YpO5/YtmrdpE5a7=97xZBiPiGaiDD; ssxmod_itna2=1-YuiQ0KBIx0ObDQG03GCmaG70q7ItKGHDyxWFK0CDmxjKidqDUnPHDiw87r_5i=9PjDE0G3hbe4iThGe_Euv=kFi5XtV4FEDwhrZ4rD',
}


for page in range(1,50):
    print(f'正在采集{page}')
    
    params = {
    'api_key': '51job',
    'timestamp': str(int(time.time())),
    'keyword': '供应链',
    'searchType': '2',
    'function': '',
    'industry': '',
    'jobArea': '000000',
    'jobArea2': '',
    'landmark': '',
    'metro': '',
    'salary': '',
    'workYear': '',
    'degree': '',
    'companyType': '',
    'companySize': '',
    'jobType': '',
    'issueDate': '',
    'sortType': '0',
    'pageNum': str(page),
    'requestId': '',
    'keywordType': '',
    'pageSize': '20',
    'source': '1',
    'accountId': '',
    'pageCode': 'sou|sou|soulb',
    'scene': '7',
}
        
    response = requests.get(
        url='https://we.51job.com/api/job/search-pc',
        params=params,
        cookies=cookies,
        headers=headers
    )
    """ print("响应状态码：", response.status_code)
    print("原始响应内容（前500字符）：", response.text[:500])  # 只打印前500字符，避免刷屏 """
    try:
        if '<script' in response.text or response.status_code != 200:
            print(f'第{page}页被反爬，尝试处理...')
            import re
            arg1 = re.search(r"var arg1='(.*?)';", response.text)
            if arg1:
                arg1 = arg1.group(1)
                def decrypt_arg1(arg1):
                    return ''.join([chr(int(arg1[i:i+2], 16) ^ 0x36) for i in range(0, len(arg1), 2)])
                acw_sc__v2 = decrypt_arg1(arg1)
                cookies['acw_sc__v2'] = acw_sc__v2
                response = requests.get(
                    url='https://we.51job.com/api/job/search-pc',
                    params=params,
                    cookies=cookies,
                    headers=headers
                )
            else:
                print(f'第{page}页反爬处理失败，跳过')
                continue  
        
        json_data = response.json()
    
        job_list = json_data.get('resultbody', {}).get('job', {}).get('items', [])
        if not job_list:
            print(f'第{page}页无数据，停止爬取')
            break 

        for job in job_list:
            job_info = {}
            area_info = job.get('jobAreaString', '').split('.')
            if len(area_info) >= 2:
                job_info['工作城市'] = area_info[0]
                job_info['工作城区'] = area_info[1]
            else:
                job_info['工作城市'] = area_info[0] if area_info else 'unknown'
                job_info['工作城区'] = 'unknown'

            job_info['职位名称'] = job.get('jobName', '')
            job_info['公司名称'] = job.get('companyName', '')
            job_info['薪资'] = job.get('provideSalaryString', '')
            job_info['发布时间'] = job.get('issueDateString', '')
            job_info['工作经验要求'] = job.get('workYearString', '')
            job_info['学历要求'] = job.get('degreeString', '')
            job_info['职位标签'] = ','.join(job.get('jobTags', []))
            job_info['职位描述'] = job.get('jobDescribe', '').replace('\n', ' ')
            job_info['公司类型'] = job.get('companyTypeString', '')
            job_info['公司规模'] = job.get('companySizeString', '')
            job_info['职位链接'] = job.get('jobHref', '')

            csv_writer.writerow(job_info)
            print(f'第{page}页：{job_info["职位名称"]} 写入成功')
        time.sleep(2)
    
    except requests.exceptions.JSONDecodeError:
        print(f'第{page}页 JSON 解析失败，响应内容：{response.text[:200]}')
        continue 
    except Exception as e:
        print(f'第{page}页出错：{str(e)}')
        continue

f.close()
print('爬取完成，数据已保存')