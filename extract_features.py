from collections import OrderedDict, Counter

import pandas as pd
import numpy as np


CLIP_LEN_MS = 60000
ANNOTATION_LEN_MS = 20000  # Granularity at which annotations were done (clip stride)
CONCURRENT = 5000  # Milliseconds


def extract_features(df, num_students, possible_events):
    # Extract features from a Pandas DataFrame, which might consist of one group for one 30-second
    # window, or a cumulative window, or something else

    df = df[df.event_type != 'accel_changed']  # Calculate accel features separately
    if 'object_top' not in df.columns:  # Small log files don't always have any edits
        df['object_top'] = df['object_left'] = df['object_path_length'] = df['object_height'] = \
            df['object_width'] = np.nan
        df['object_type'] = ''
    result = OrderedDict({
        'num_rows': len(df),
        'prop_students_acting': len(df.student_id.unique()) / num_students,
        'prop_students_editing': len(df[df.object_top.notnull()]
                                     .student_id.unique()) / num_students,
        'prop_students_scrolling': len(df[df.event_type == 'scrollposition_changed']
                                       .student_id.unique()) / num_students,
        'num_unique_pages_viewed': len(df.page.unique()),
    })

    # Event type counts
    for event_type in sorted(possible_events):
        result['num_' + event_type] = sum(df.event_type == event_type)

    # Features past this point do not make sense without data (missing values are semantic, not 0)
    if not len(df):
        return result

    # Calculate per-student values
    dist_scrolled = []
    num_actions = []
    num_edits = []
    mean_dist_between_edits = []
    scroll_pos_stddev = []
    for pid in df.student_id.unique():
        pid_df = df[df.student_id == pid]
        scroll_events = pid_df[(pid_df.event_type == 'scrollposition_changed')]
        dist_scrolled.append(sum(abs(a - b) for a, b in zip(
            scroll_events.scroll_pos.values[1:], scroll_events.scroll_pos.values[:-1])))
        num_actions.append(len(pid_df))
        num_edits.append(sum(pid_df.object_top.notnull()))
        edits = pid_df[pid_df.object_top.notnull()]
        edits_dist = np.sqrt((edits.object_top - edits.shift(1).object_top) ** 2 +
                             (edits.object_left - edits.shift(1).object_left) ** 2)
        mean_dist_between_edits.append(edits_dist.mean())
        scroll_pos_stddev.append(np.mean([scroll_events[scroll_events.page == p].scroll_pos.std()
                                          for p in scroll_events.page.unique()]))
    num_actions = sorted(num_actions)  # Sort by activity
    num_edits = sorted(num_edits)
    result['dist_drawn'] = df[(df.event_type == 'add_object') & (df.object_type == 'path')] \
        .object_path_length.sum()
    result['dist_scrolled'] = sum(dist_scrolled)
    result['scroll_stddev'] = np.nanmean(scroll_pos_stddev)
    result['ratio_least_to_most_active'] = num_actions[0] / num_actions[-1]
    result['ratio_2ndmost_to_most_active'] = \
        num_actions[-2] / num_actions[-1] if len(num_actions) > 1 else 0
    result['ratio_least_to_2ndleast_active'] = \
        num_actions[0] / num_actions[1] if len(num_actions) > 1 else 0
    result['ratio_2ndmost_edits_to_most_edits'] = \
        num_edits[-2] / num_edits[-1] if len(num_edits) > 1 and num_edits[-1] > 0 else 0
    result['mean_dist_same_student_edits'] = np.nanmean(mean_dist_between_edits)

    # Calculate values that require iterating over all events
    cur_page_map = {}  # pid -> page num
    result['prop_diff_pages_max'] = 0
    result['prop_diff_pages_min'] = 1
    cur_scroll_pos_map = {}  # pid -> scroll pos
    result['min_scroll_pos_diff'] = 999999999
    result['max_scroll_pos_diff'] = -1
    last_editor_id = None
    cons_edits_dist = []  # Distances between consecutive edits from different students (same page)
    cons_edits_diffpage = []  # Same/different page for consecutive edits from different students
    for _, row in df.iterrows():
        cur_page_map[row.student_id] = row.page
        prop_diff_pages = len(set(cur_page_map.values())) / num_students
        result['prop_diff_pages_max'] = max(result['prop_diff_pages_max'], prop_diff_pages)
        result['prop_diff_pages_min'] = min(result['prop_diff_pages_min'], prop_diff_pages)
        cur_scroll_pos_map[row.student_id] = row.scroll_pos  # Or scroll_view_top
        for pida in cur_scroll_pos_map:
            for pidb in cur_scroll_pos_map:
                if pida != pidb and cur_page_map[pida] == cur_page_map[pidb]:
                    result['min_scroll_pos_diff'] = min(
                        result['min_scroll_pos_diff'],
                        abs(cur_scroll_pos_map[pida] - cur_scroll_pos_map[pidb]))
                    result['max_scroll_pos_diff'] = max(
                        result['max_scroll_pos_diff'],
                        cur_scroll_pos_map[pida] - cur_scroll_pos_map[pidb])
        if not np.isnan(row.object_top):  # Is editing action
            if last_editor_id is not None and last_editor_id != row.student_id:
                if cur_page_map[last_editor_id] == row.page:
                    cons_edits_dist.append(np.sqrt((last_edit_pos_top - row.object_top) ** 2 +
                                                   (last_edit_pos_left - row.object_left) ** 2))
                    cons_edits_diffpage.append(0)  # Editing on same page
                else:
                    cons_edits_diffpage.append(1)  # Editing on different pages
            last_edit_pos_top = row.object_top
            last_edit_pos_left = row.object_left
            last_editor_id = row.student_id
    if result['max_scroll_pos_diff'] < 0:
        result['max_scroll_pos_diff'] = result['min_scroll_pos_diff'] = np.nan
    result['mean_dist_consecutive_edits'] = np.mean(cons_edits_dist)
    result['prop_consecutive_edits_diffpages'] = np.mean(cons_edits_diffpage)
    result['max_sec_no_actions'] = max((df.timestamp.shift(-1) - df.timestamp).values) / 1000

    # Calculate "concurrent" features by iterating over short windows of time
    '''These are disabled for now because they don't help much and are very slow to extract.
    editing_pos_diffs = []
    num_windows = 0
    num_win_no_actions = 0
    num_win_all_scrolling = 0
    num_win_edits = 0
    result['max_prop_cc_editing'] = 0
    result['mean_prop_cc_editing'] = 0
    for start_time_ms in range(df.timestamp.iloc[0], df.timestamp.iloc[-1] + 1, CONCURRENT):
        win_df = df[(df.timestamp >= start_time_ms) & (df.timestamp < start_time_ms + CONCURRENT)]
        editing_pos_top = []
        editing_pos_left = []
        editing_pos_diffpage = []
        editing_page = []
        for pid in win_df.student_id.unique():
            pid_edits = win_df[(win_df.student_id == pid) & win_df.object_top.notnull()]
            if len(pid_edits):
                editing_page.append(pid_edits.page.iloc[0])
                editing_pos_top.append(pid_edits.object_top.iloc[0])  # Just take first one
                editing_pos_left.append(pid_edits.object_left.iloc[0])
        for i, page in enumerate(editing_page):
            for j, page2 in enumerate(editing_page):
                if i != j and page == page2:
                    editing_pos_diffs.append(
                        np.sqrt((editing_pos_top[i] - editing_pos_top[j]) ** 2 +
                                (editing_pos_left[i] - editing_pos_left[j]) ** 2))
                    editing_pos_diffpage.append(0)  # Editing on same page
                elif i != j:
                    editing_pos_diffpage.append(1)  # Editing on different pages
        num_windows += 1
        num_win_no_actions += 0 if len(win_df) else 1
        num_win_all_scrolling += 1 if len(win_df[win_df.event_type == 'scrollposition_changed']
                                          .student_id.unique()) == num_students else 0
        num_win_edits += 1 if sum(win_df.object_top.notnull()) else 0
        prop_cc_edit = len(win_df[win_df.object_top.notnull()].student_id.unique()) / num_students
        result['max_prop_cc_editing'] = max(result['max_prop_cc_editing'], prop_cc_edit)
        result['mean_prop_cc_editing'] += prop_cc_edit
    result['mean_cc_editing_pos_diff'] = np.mean(editing_pos_diffs)
    result['max_cc_editing_pos_diff'] = \
        np.max(editing_pos_diffs) if len(editing_pos_diffs) else np.nan
    result['min_cc_editing_pos_diff'] = \
        np.min(editing_pos_diffs) if len(editing_pos_diffs) else np.nan
    result['prop_cc_edits_diffpages'] = np.mean(editing_pos_diffpage)
    result['prop_cc_periods_no_actions'] = num_win_no_actions / num_windows
    result['prop_cc_periods_all_scrolling'] = num_win_all_scrolling / num_windows
    result['prop_cc_periods_edits'] = num_win_edits / num_windows
    result['mean_prop_cc_editing'] /= num_windows
    '''

    result['edit_bbox_min_x'] = df.object_left.min()
    result['edit_bbox_max_x'] = (df.object_left + df.object_width).max()
    result['edit_bbox_min_y'] = df.object_top.min()
    result['edit_bbox_max_y'] = (df.object_top + df.object_height).max()
    result['edit_mean_x'] = (df.object_left + df.object_width / 2).mean()
    result['edit_mean_y'] = (df.object_top + df.object_height / 2).mean()

    return result


if __name__ == '__main__':
    print('Loading data')
    INPUT_FILE = 'data/csteps2_2017-12-05.csv'  # head10k.csv or data/csteps2_2017-12-05.csv
    all_df = pd.read_csv(INPUT_FILE)
    all_df = all_df[all_df.student_type == 'student']  # Remove any TAs who may join
    group_map = pd.read_csv('data/csteps2_2017-12-05_netid_group_mapping.csv')
    group_map = {r.NetID: r.Group for _, r in group_map.iterrows()}
    possible_events = all_df.event_type.unique()

    # Mapping of day + class ID to video start Unix timestamps
    video_start = {
        '2017-09-12 ADA_recovered': 1505239161,
        '2017-09-12 ADB': 1505242760,
        '2017-09-12 ADC': 1505246426,
        '2017-09-12 ADD': 1505250000,
        '2017-09-26 ADA_failed': 1506448770,
        '2017-09-26 ADB': 1506452482,
        '2017-09-26 ADC_extra_sheet': 1506456000,
        '2017-09-26 ADD': 1506459597,
        '2017-10-17 ADA': 1508263208,
        '2017-10-17 ADB': 1508266825,
        '2017-10-17 ADC': 1508270404,
        '2017-10-17 ADD': 1508274001,
        '2017-10-31 ADA': 1509472800,
        '2017-10-31 ADB': 1509476893,  # 8:13 into the hour--unusually late
        '2017-10-31 ADC': 1509480044,
        '2017-10-31 ADD': 1509483600,
        '2017-11-07 ADA': 1510081200,
        '2017-11-07 ADB': 1510084572,  # A bit unusually early, ~4 minutes
        '2017-11-07 ADC': 1510088324,
        '2017-11-07 ADD': 1510091906,
        '2017-11-28 ADA': 1511895600,
        '2017-11-28 ADB': 1511899200,
        '2017-11-28 ADC': 1511902812,
        '2017-11-28 ADD': 1511906400,
        '2017-12-05 ADA': 1512500920,  # 8:40 into the hour
        '2017-12-05 ADB': 1512504483,  # 8:03 into the hour
        '2017-12-05 ADC': 1512508289,  # 11:29 into the hour
        '2017-12-05 ADD': 1512512012,  # 13:32 into the hour
    }

    instances = []
    for group_filename in sorted(all_df.source_file.unique()):
        print('Processing ' + group_filename)
        group_df = all_df[all_df.source_file == group_filename]
        uniq_students = [p.lower().replace('@illinois.edu', '') for p in group_df.netid.unique()]
        groupnum = Counter(group_map[p] for p in uniq_students).most_common()
        # if len(groupnum) > 1 and '?' in [g for g, _ in groupnum]:
        #     print(groupnum)
        #     print(uniq_students)
        if len(groupnum) > 1 and groupnum[0][1] == groupnum[1][1]:  # Ambiguous group number
            print(groupnum)
            print(uniq_students)
            groupnum = ';'.join('G' + str(g[0]) for g in groupnum)
        else:
            groupnum = 'G' + str(groupnum[0][0])

        clipping_start_ms = group_df.timestamp.iloc[0]
        try:  # Back up clipping process to align to video start modulo clip length
            vid_start_ms = \
                video_start[group_df.day.iloc[0] + ' ' + group_df.class_id.iloc[0]] * 1000
            if vid_start_ms < clipping_start_ms:  # Vid starts first
                clipping_start_ms = vid_start_ms - CLIP_LEN_MS + ANNOTATION_LEN_MS
            else:  # Clip starts first
                clipping_start_ms -= (clipping_start_ms - vid_start_ms) % CLIP_LEN_MS
                clipping_start_ms += ANNOTATION_LEN_MS
        except KeyError:
            vid_start_ms = False

        for start_ms in range(clipping_start_ms, group_df.timestamp.iloc[-1] + 1,
                              ANNOTATION_LEN_MS):
            clip_df = group_df[(group_df.timestamp >= start_ms) &
                               (group_df.timestamp < start_ms + CLIP_LEN_MS)]
            instances.append(OrderedDict({
                'day': group_df.day.iloc[0],
                'class_id': group_df.class_id.iloc[0],
                'source_file': group_filename,
                'group_num': groupnum,
                'clip_start_ms': start_ms,
                'clip_end_ms': start_ms + CLIP_LEN_MS - 1,
                'start_sec_into_video': (start_ms - vid_start_ms) / 1000 if vid_start_ms else '',
                'start_sec_into_log': (start_ms - group_df.timestamp.iloc[0]) / 1000,
            }))
            instances[-1].update(extract_features(clip_df, len(uniq_students), possible_events))
            cum_df = group_df[group_df.timestamp < start_ms + CLIP_LEN_MS]
            for k, v in extract_features(cum_df, len(uniq_students), possible_events).items():
                instances[-1]['cum_' + k] = v

    print('Saving')
    pd.DataFrame.from_records(instances).to_csv(INPUT_FILE[:-4] + '_features.csv', index=False)
