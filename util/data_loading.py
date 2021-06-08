import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from typing import List
from typing import Union
from oratio.Session import Session
from oratio.processing.MouseProcessor import MouseProcessor
from oratio.processing.KeyboardProcessor import KeyboardProcessor
from oratio.processing.SessionMetaProcessor import SessionMetaProcessor


def load_db(path: str, channel: str):
    query = 'SELECT \
              Session.session,\
              Session.time, '
    if channel != "Camera":
        query += f'{channel}RawData.data AS data, '
    else:
        query += 'Camera.Images AS data, '

    query += 'MetaRawData.data AS MetaRawData, \
              Session.label\
            FROM\
              Session '
    if channel != "Camera":
        query += f'JOIN {channel}RawData ON CAST(Session.session AS INT) = {channel}RawData.session '
    else:
        query += 'JOIN Camera ON CAST(Session.session AS INT) = Camera.session '

    query += 'JOIN MetaRawData ON MetaRawData.session = CAST(Session.session AS INT)'
    with sqlite3.connect(path) as connection:
        db = pd.read_sql_query(query, connection)

    db["data"] = [np.array(pickle.loads(d)) for d in db["data"]]
    db["MetaRawData"] = [np.array(pickle.loads(d)) for d in db["MetaRawData"]]
    return db


def fix_time_margins(db, margin: float = 2, fix_data=True):
    session_times = db["time"].values
    for index in range(1, len(session_times)):
        dt = session_times[index] - session_times[index - 1]
        if dt > margin:
            session_times[index:] -= dt - margin
            if fix_data:
                __update_event_times(db["data"][index:], dt - margin)

    db["time"] = session_times


def __update_event_times(events, dt: float):
    for session in events:
        for event in session:
            event.Timestamp -= dt


def merge_sessions(df, num_sessions):
    merged_df = pd.DataFrame(columns=df.columns.to_list())
    prev_label = {'categorical': 'none',
                  'VAD': {'valance': 4,
                          'arousal': 4,
                          'dominance': 4}}
    for sid, i in enumerate(range(0, len(df), num_sessions)):
        meta_data = list()
        data = list()
        categorical = {"happiness": 0,
                       "disgust": 0,
                       "fear": 0,
                       "sadness": 0,
                       "anger": 0,
                       "surprise": 0,
                       'none': 0}
        valance = 0
        arousal = 0
        dominance = 0

        for j in range(num_sessions):
            if i + j < len(df):
                data.extend(df["data"][i + j])
                meta_data.extend(df["MetaRawData"][i + j])

                label = json.loads(df["label"][i + j].replace("'", "\""))

                if type(label) != dict:
                    label = prev_label
                if "categorical" not in label or label["categorical"] == 'none':
                    label["categorical"] = prev_label["categorical"]
                if "VAD" not in label:
                    label["VAD"] = prev_label["VAD"]
                if "valance" not in label['VAD'] or label['VAD']["valance"] == 'none':
                    label['VAD']["valance"] = prev_label['VAD']["valance"]
                if "arousal" not in label['VAD'] or label['VAD']["arousal"] == 'none':
                    label['VAD']["arousal"] = prev_label['VAD']["arousal"]
                if "dominance" not in label['VAD'] or label['VAD']["dominance"] == 'none':
                    label['VAD']["dominance"] = prev_label['VAD']["dominance"]
                prev_label = label

                categorical[label["categorical"]] += 1
                valance += label['VAD']['valance']
                arousal += label['VAD']['arousal']
                dominance += label['VAD']['dominance']
            else:
                j -= 1
                break

        max_categorical_count = max(list(categorical.values()))
        max_categorical_index = list(categorical.values()).index(max_categorical_count)
        max_categorical = list(categorical.keys())[max_categorical_index]

        label = {'categorical': max_categorical,
                 'VAD': {'valance': valance / (j + 1),
                         'arousal': arousal / (j + 1),
                         'dominance': dominance / (j + 1)}}
        merged_df = merged_df.append({'session': sid,
                                      'time': df["time"][i],
                                      'data': data,
                                      'MetaRawData': meta_data,
                                      'label': label}, ignore_index=True)
    return merged_df


def process_raw_data(db, duration: int, channel: str, metadata_model_path: str):
    if channel == "Keyboard":
        processor = KeyboardProcessor()
        init_features = processor._KeyboardProcessor__init_features
    elif channel == "Mouse":
        processor = MouseProcessor()
        init_features = processor._MouseProcessor__init_features
    elif channel == "Camera":
        db = db.rename(columns={"data": "images"})
        db = db.assign(data=np.nan)
    metadata_processor = SessionMetaProcessor(resources_path=metadata_model_path)

    for sid in range(len(db)):
        session = Session(sid, duration, None, None)
        session.start_time = db['time'][sid]

        if channel != "Camera":
            data = db["data"][sid]
            features = processor.process_data(data, session)
            if features["idle_time"] >= duration:
                # hack to call a private method of a class
                init_features(session)
                features = processor.features

            for feature in features:
                db.loc[sid, feature] = features[feature]

        metadata = db["MetaRawData"][sid]

        features = metadata_processor.process_data(metadata, session)
        for feature in features:
            db.loc[sid, feature] = features[feature]

    db = db.drop(labels=["data", "MetaRawData"], axis=1)
    return db


def load_features(path: str, channels: Union[List[str], str], duration: int, metadata_model_path: str = "/content/drive/MyDrive/capi/models"):
    if type(channels) == str:
        channels = [channels]
    data = None
    for channel in channels:
        db = load_db(path, channel)
        fix_data = channel != "Camera"
        fix_time_margins(db, margin=1.5, fix_data=fix_data)
        mdb = merge_sessions(db, duration)
        pdb = process_raw_data(mdb, duration, channel, metadata_model_path)
        pdb = pdb.rename(columns={"idle_time": "idle_time_"+channel})
        if data is None:
            data = pdb
        data = data.join(pdb[pdb.columns.difference(data.columns)])

    categorical_list = []
    valance_list = []
    arousal_list = []
    dominance_list = []

    for index, row in data.iterrows():
        categorical_list.append(row["label"]["categorical"])
        valance_list.append(row["label"]["VAD"]["valance"])
        arousal_list.append(row["label"]["VAD"]["arousal"])
        dominance_list.append(row["label"]["VAD"]["dominance"])

    data = data.drop(columns=["label"])
    data["categorical"] = categorical_list
    data["valance"] = valance_list
    data["arousal"] = arousal_list
    data["dominance"] = dominance_list
    data["positive"] = (data["valance"] >= 4).astype(int)
    return data


def load_all(paths: List[str], channels: Union[List[str], str], session_length=10, use_pooling=True):
    dbs = [p for p in paths]
    if use_pooling:
        from multiprocessing import Pool
        import itertools
        p = Pool()
        dbs = p.starmap(load_features, zip(dbs, itertools.repeat(channels, len(dbs)), itertools.repeat(session_length, len(dbs))))
    else:
        dbs = [load_features(name, channels, session_length) for name in dbs]
    return dbs