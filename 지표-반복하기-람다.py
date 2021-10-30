import pandas as pd
import boto3
import joblib

def get_action_iteration_score(Bucket, FileKey):
    score = 0
    appeared_actions = {}

    df = pd.read_csv(Bucket.Object(FileKey).get()['Body'])

    for str_actions_with_confs in df['actions']:
        actions_with_confs = eval(str_actions_with_confs)

        for action_with_conf in actions_with_confs:
            action = action_with_conf['action']

            if action in appeared_actions:
                score += 1/appeared_actions[action]

            appeared_actions[action] = 0

        for action in appeared_actions.keys():
            appeared_actions[action] += 1

    return score

if __name__ == '__main__':
    S3_ACCESS_KEY_ID = '...'
    S3_SECRET_ACCESS_KEY = '...'
    BUCKET_NAME = '...'

    s3 = boto3.resource(
        's3',
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY
    )
    s3_bucket = s3.Bucket(name=BUCKET_NAME)

    score = get_action_iteration_score(s3_bucket, '37aok0iq0rqpnqv5uadj0evq8h/37aok0iq0rqpnqv5uadj0evq8h_1_C.csv')
    print(f'score: {score}')
    
    model = joblib.load('./지표모델/지표-반복하기-회귀-모델.pkl')
    y_predicted = model.predict([[score]])

    print(f'y_predicted: {y_predicted}')

    print('Program terminated successfully.')
