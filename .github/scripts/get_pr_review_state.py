import requests
import json
from concurrent.futures import ThreadPoolExecutor

# GitHub Personal Access Token (PAT)
GITHUB_TOKEN = "ghp_0000"  # github token 입력 필요
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

# 리포지토리 정보
REPO_OWNER = "LangChain-OpenTutorial"  # 리포지토리 소유자
REPO_NAME = "LangChain-OpenTutorial"  # 리포지토리 이름

# PR 데이터 가져오기
def fetch_pull_requests():
    pr_data = []
    page = 1
    while True:
        pulls_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls?state=all&per_page=100&page={page}"
        response = requests.get(pulls_url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to fetch PRs. Status Code: {response.status_code}")
            break

        page_data = response.json()
        if not page_data:  # 더 이상 데이터가 없으면 종료
            break

        pr_data.extend(page_data)
        page += 1

    return pr_data


# 리뷰 데이터 가져오기 (리뷰어별 정리)
def fetch_reviews(pr):
    pr_number = pr["number"]
    pr_author = pr["user"]["login"]
    pr_title = pr["title"]
    pr_created_at = pr["created_at"]
    pr_updated_at = pr["updated_at"]
    pr_state = pr["state"]  # PR 상태

    # 리뷰 데이터 가져오기
    reviews_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/reviews"
    review_response = requests.get(reviews_url, headers=headers)

    if review_response.status_code != 200:
        print(f"Failed to fetch reviews for PR #{pr_number}. Status Code: {review_response.status_code}")
        return [{
            "PR Number": pr_number,
            "PR Submitter ID": pr_author,
            "PR Title": pr_title,
            "PR Create Time": pr_created_at,
            "PR Last Update Time": pr_updated_at,
            "PR State (now)": pr_state,
            "Reviewer ID": None,
            "Review State (now)": None,
            "Review Create Time": None
        }]

    reviews = review_response.json()

    # 리뷰어별 데이터 정리
    reviewer_data = {}
    for review in reviews:
        reviewer = review["user"]["login"]
        review_state = review["state"]
        review_submitted_at = review["submitted_at"]

        # 리뷰어별로 최초 시간과 최종 상태를 기록
        if reviewer not in reviewer_data:
            reviewer_data[reviewer] = {
                "Review Create Time": review_submitted_at,
                "Review State (now)": review_state
            }
        else:
            # 최신 상태만 업데이트
            reviewer_data[reviewer]["Review State (now)"] = review_state

    # 리뷰어별 데이터를 테이블 형태로 변환
    result = []
    for reviewer, data in reviewer_data.items():
        result.append({
            "PR Number": pr_number,
            "PR Submitter ID": pr_author,
            "PR Title": pr_title,
            "PR Create Time": pr_created_at,
            "PR Last Update Time": pr_updated_at,
            "PR State (now)": pr_state,
            "Reviewer ID": reviewer,
            "Review State (now)": data["Review State (now)"],
            "Review Create Time": data["Review Create Time"]
        })

    # 리뷰가 없는 경우 처리
    if not result:
        result.append({
            "PR Number": pr_number,
            "PR Submitter ID": pr_author,
            "PR Title": pr_title,
            "PR Create Time": pr_created_at,
            "PR Last Update Time": pr_updated_at,
            "PR State (now)": pr_state,
            "Reviewer ID": None,
            "Review State (now)": None,
            "Review Create Time": None
        })

    return result


# 병렬 처리로 리뷰 데이터 가져오기
def fetch_all_reviews_parallel(pr_data):
    result_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # PR 데이터에 대해 병렬로 리뷰 데이터 수집
        futures = executor.map(fetch_reviews, pr_data)
        for future in futures:
            if future:  # 리뷰 데이터가 있으면 추가
                result_data.extend(future)
    return result_data


import pandas as pd
from datetime import datetime, timedelta

# UTC -> KST 변환 함수
def convert_to_kst(utc_time_str):
    if utc_time_str is None:
        return None
    utc_time = datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%SZ")  # UTC 시간으로 변환
    kst_time = utc_time + timedelta(hours=9)  # 9시간 더하기
    return kst_time.strftime("%Y-%m-%d %H:%M:%S")  # 원하는 포맷으로 변환

# main
if __name__ == "__main__":

    print("Fetching pull requests...")
    pr_data = fetch_pull_requests()

    print("Fetching reviews in parallel...")
    result_data = fetch_all_reviews_parallel(pr_data)

    # 결과를 JSON 파일로 저장
    with open("pr_review_data.json", "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    df = pd.read_json("pr_review_data.json")
    print("Data saved to pr_review_data.json")
    
    df["PR Create Time (KST)"] = df["PR Create Time"].apply(convert_to_kst)
    df["PR Last Update Time (KST)"] = df["PR Last Update Time"].apply(convert_to_kst)
    df["Review Create Time (KST)"] = df["Review Create Time"].apply(convert_to_kst)
    
    df.to_csv("pr_review_data_cleaned.csv", encoding='utf-8-sig')
    print(df.head())
    
    print("Data saved to pr_review_data_cleaned.csv")
