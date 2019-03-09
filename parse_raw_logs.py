import json
import os
from collections import OrderedDict

import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_line(line, source_file, studentid_netid_map, object_data):
    # Parses one line of JSON data from a log file.
    row = OrderedDict()
    try:
        obj = json.loads(line)
    except json.decoder.JSONDecodeError:
        return None  # Some logs were corrupt and incomplete (8/29 and 9/12/2017).
    assert 'updates' in obj  # Otherwise what is the meaning of this line?
    for key in obj['updates']:
        if key.startswith('latestObjects.'):
            continue  # Weirdness in log_350, maybe others, where this needs to be ignored.
        updates = obj['updates'][key]
        data = json.loads(updates['data']) if 'data' in updates else None
        meta = json.loads(updates['meta']) if 'meta' in updates else None
        row['source_file'] = source_file
        row['timestamp'] = obj['time']
        row['student_id'] = str(key.split('.')[1]) if '.' in key else ''
        if key == 'membershipChange':
            row['student_id'] = str(updates['id'])
            row['event_type'] = updates['action'].replace(' ', '_')
            if 'reload' in row['event_type']:
                row['event_type'] = 'leave_reload_app'
            elif 'session_ended' in row['event_type']:
                row['event_type'] = 'leave_session_ended'
            row['student_type'] = \
                {0: 'admin', 1: 'teacher', 2: 'student'}[updates['type']]
            row['num_objects_selected'] = 0  # Reset.
            row['tool_id'] = 'draw'
            assert row['student_id'] not in studentid_netid_map or \
                studentid_netid_map[row['student_id']] == updates['email']
            studentid_netid_map[row['student_id']] = updates['email']
        elif key.startswith('accel.'):
            row['event_type'] = 'accel_changed'
            row['accel_x'] = updates['x']
            row['accel_y'] = updates['y']
            row['accel_z'] = updates['z']
            row['accel_alpha'] = updates['a']
            row['accel_beta'] = updates['b']
            row['accel_gamma'] = updates['g']
        elif key.startswith('objects'):
            row['student_id'] = str(meta['u'])
            row['object_id'] = key[key.index('.') + 1:]

            if meta['type'] in ['addFreeDrawing', 'addFBDObject']:
                row['event_type'] = 'add_object'
                object_data[row['object_id']] = {'type': data['type']}
            elif meta['type'] in ['modifyObject', 'modifyFBDObject']:
                row['event_type'] = 'modify_object'
            elif meta['type'] == 'removeObject':
                row['event_type'] = 'remove_object'
            elif meta['type'] == 'undoAddObject':
                row['event_type'] = 'undo_add_object'
            elif meta['type'] == 'undoRemoveObject':
                row['event_type'] = 'undo_remove_object'
            elif meta['type'] == 'undoModifyObject':
                row['event_type'] = 'undo_modify_object'
            else:
                raise ValueError('Unknown value of "objects" event_type')
            row['object_type'] = data['type'] if 'type' in data else \
                object_data[row['object_id']]['type']

            if 'top' in data and data['top'] is not None:  # Record size/position.
                data['width'] = data['width'] if data['width'] else 0
                data['height'] = data['height'] if data['height'] else 0
                row['object_left'] = data['left'] - data['width'] / 2
                row['object_top'] = data['top'] - data['height'] / 2
                row['object_width'] = data['width']
                row['object_height'] = data['height']
                row['object_scale_x'] = data['scaleX']
                row['object_scale_y'] = data['scaleY']
                row['object_rotation_angle'] = data['angle']
                if 'path' in data:
                    row['object_num_points'] = len(data['path'])
                    if len(data['path']) == 2 and not data['path'][1][1]:
                        data['path'][1][1] = data['path'][0][1]
                        data['path'][1][2] = data['path'][0][2]
                    row['object_path_length'] = sum(
                        np.sqrt(((p[3] - p[1]) ** 2) + ((p[4] - p[2]) ** 2))
                        for p in data['path'] if len(p) > 4 and p[0] == 'Q')  # Q=SVG point.
                # Compare object geometry to discover moves/rotations etc.
                prev = object_data[row['object_id']]
                if 'object_left' in prev and row['event_type'] == 'modify_object':
                    if prev['object_rotation_angle'] != row['object_rotation_angle']:
                        row['event_type'] = 'rotate_object'
                    elif prev['object_scale_x'] != row['object_scale_x'] or \
                            prev['object_scale_y'] != row['object_scale_y']:
                        row['event_type'] = 'resize_object'
                    elif prev['object_top'] != row['object_top'] or \
                            prev['object_left'] != row['object_left']:
                        row['event_type'] = 'move_object'
                for k in row:  # Update object geometry for later comparisons.
                    if k.startswith('object_'):
                        object_data[row['object_id']][k] = row[k]

            if row['event_type'] == 'remove_object':
                # Add in position/scale/etc. at time of removal.
                prev = object_data[row['object_id']]
                for k in prev:
                    if k.startswith('object_'):
                        row[k] = prev[k]

        elif key.startswith('penColor'):
            row['event_type'] = 'pen_color_set'
        elif key.startswith('scrollPositions'):
            row['event_type'] = 'scrollposition_changed'
            row['scroll_pos'] = updates['pos']
            row['scroll_view_top'] = updates['viewTop']
            row['scroll_view_bottom'] = updates['viewBottom']
        elif key.startswith('selectionBox'):
            row['event_type'] = 'selection_changed'
            if int(updates['visible']):
                row['selection_x'] = updates['left']
                row['selection_y'] = updates['top']
                row['selection_width'] = updates['width']
                row['selection_height'] = updates['height']
                row['num_objects_selected'] = len(updates['contents'])
        elif key.startswith('setPage'):
            row['event_type'] = 'set_page'
            row['student_id'] = str(meta['u'])
            row['page'] = int(data[row['student_id']])
        elif key.startswith('tool'):
            row['event_type'] = 'tool_changed'
            row['tool_id'] = {0: 'draw', 2: 'erase', 3: 'select'}[int(updates['tool'])]
        elif key.startswith('userColors'):
            row['event_type'] = 'user_color_set'
        else:
            raise ValueError('Unknown key in JSON string: ' + str(key))
    return row


def append_row(row, existing_rows):
    # Add a new row to the existing rows, filling in unknown values where needed.
    ffill_cols = ['accel_x', 'accel_y', 'accel_z', 'accel_alpha', 'accel_beta', 'accel_gamma',
                  'num_objects_selected', 'tool_id', 'student_type', 'page']
    scroll_cols = ['scroll_pos', 'scroll_view_top', 'scroll_view_bottom']
    if row is not None:
        # Forward fill missing values for appropriate fields.
        for col in ffill_cols:
            if col not in row:  # Need to fill.
                for prev_row in reversed(existing_rows):
                    if prev_row['student_id'] == row['student_id']:
                        if col in prev_row:  # Found the previous value for this student.
                            row[col] = prev_row[col]
                        break
        # Scroll position columns are handled differently because they are page-specific.
        for col in scroll_cols:
            if col not in row:
                for prev_row in reversed(existing_rows):
                    if prev_row['student_id'] == row['student_id'] and \
                            prev_row['page'] == row['page']:
                        if col in prev_row:
                            row[col] = prev_row[col]
                        break
        if 'page' not in row:
            row['page'] = 0  # Initial page number when joining.
        existing_rows.append(row)


def parse_logfile(filename):
    rows = []
    studentid_netid_map = {}
    object_data = {}  # Track some object data to distinguish move/scale/etc. events.
    with open(filename) as infile:
        for line in tqdm(infile):
            row = parse_line(line, filename.split('/')[-1], studentid_netid_map, object_data)
            append_row(row, rows)

    df = pd.DataFrame.from_records(rows)
    # # Copy values forward and backward for appropriate fields, where they are blank.
    # cols = [c for c in ffill_cols if c in df.columns]  # In case some event types never happened.
    # if 'page' not in df.columns:
    #     df['page'] = 0  # Apparently not all users immediately start with a set_page event.
    # for pid in df.student_id.unique():
    #     pid_i = df.student_id == pid
    #     df.loc[pid_i, cols] = df[pid_i][cols].fillna(method='ffill')
    #     # Backfill is just for accelerometer data. However, it is very inconvenient to have to
    #     # backfill in real-time applications, and not that important, so we'll 0-fill instead.
    #     # df.loc[pid_i, 'page'] = df[pid_i].page.fillna(0)  # This does need to be 0-filled.
    #     # df.loc[pid_i, cols] = df[pid_i][cols].fillna(method='bfill')
    #     df.loc[pid_i, cols] = df[pid_i][cols].fillna(0)
    #     for page in df[pid_i].page.unique():
    #         df.loc[pid_i & (df.page == page), scroll_cols] = \
    #             df[pid_i & (df.page == page)][scroll_cols].fillna(method='ffill').fillna(0)

    return df, studentid_netid_map


if __name__ == '__main__':
    base_dir = 'data/'
    dfs = []
    studentid_netid_map = {}
    for daydir in os.listdir(base_dir):
        if not os.path.isdir(base_dir + daydir):
            continue
        for classdir in os.listdir(base_dir + daydir):
            if not os.path.isdir(base_dir + daydir + '/' + classdir):
                continue
            for fname in os.listdir(base_dir + daydir + '/' + classdir):
                if fname.startswith('log') and fname.endswith('.txt') or fname.endswith('.logfile'):
                    print(daydir + '/' + classdir + '/' + fname)
                    df, id_map = parse_logfile(base_dir + daydir + '/' + classdir + '/' + fname)
                    df.insert(0, 'day', daydir)
                    df.insert(1, 'class_id', classdir)
                    dfs.append(df)
                    assert all(i not in studentid_netid_map or studentid_netid_map[i] == id_map[i]
                               for i in id_map)  # Check if student IDs stay consistent.
                    studentid_netid_map.update(id_map)

    print('Concatenating dataframes')
    dfs = pd.concat(dfs)[dfs[0].columns]
    dfs.insert(2, 'netid', [studentid_netid_map[i] for i in dfs.student_id.values])

    print('Rescaling scroll positions for pages with positions outside [0, 1]')
    dfs['week_page'] = (dfs.timestamp / 1000 / 60 / 60 / 24 / 7).astype(int).astype(str) + \
        '_' + dfs.page.astype(str)
    for week_page in dfs.week_page.unique():
        print('Week+page: ' + week_page)
        for col in ['scroll_pos', 'scroll_view_top', 'scroll_view_bottom']:
            dfs.loc[dfs.week_page == week_page, col] /= dfs[dfs.week_page == week_page][col].max()
    dfs.drop(columns='week_page', inplace=True)

    print('Saving')
    dfs.to_csv('data/csteps2_2017-12-05.csv', index=False)
    # dfs[dfs.event_type == 'accel_changed'].dropna(axis=1, how='all') \
    #     .to_csv('data/csteps2_accel_2017-12-05.csv', index=False)  # Save only accel data.
    pd.DataFrame(list(studentid_netid_map.items()), columns=['student_id', 'email']) \
        .to_csv('data/csteps2_id_map.csv', index=False)
