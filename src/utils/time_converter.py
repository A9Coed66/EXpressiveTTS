# TODO: Phải đọc kĩ hơn về file này trong tương lai để hiểu kĩ hơn
import librosa
import datetime

def second_to_samples(s, sr):
    return librosa.time_to_samples(s, sr=sr)

def samples_to_second(d, sr):
    return librosa.samples_to_time(d, sr=sr)

def time_to_second(t: datetime.time):
    return t.hour*3600 + t.minute*60 + t.second + t.microsecond/1000000

def second_to_time(s: float):
    return datetime.time(int(s/3600), int(s/60)%60, int(s)%60, int(round(s-int(s), 3)*1000)*1000)

def time_to_samples(t: datetime.time, sr):
    return second_to_samples(time_to_second(t), sr)

def samples_to_time(d, sr):
    return second_to_time(samples_to_second(d, sr))

def time_to_name_str(t: datetime.time):
    # return f"{t.hour:02d}_{t.minute:02d}_{t.second:02d}_{(t.microsecond//1000):03d}" #06d
    return str(t)

def time_str_to_time(time_str):
    lst = time_str.split(".")
    if len(lst) == 2:
        hms, mls = lst
        if len(mls) == 3:
            mls = int(mls) * 1000
        else:
            mls = int(mls)
    else:
        hms = lst[0]
        mls = 0

    h, m, s = hms.split(':')
    h, m, s = int(h), int(m), int(s)

    return datetime.time(hour=h, minute=m, second=s, microsecond=mls)


def parse_segment_name(name, ret_time_type="str", sr=None):
    prefix, start, end = name.split("-")
    match ret_time_type:
        case "str":
            return prefix, start, end
        case "time":
            return prefix, time_str_to_time(start), time_str_to_time(end)
        case "second":
            return prefix, time_to_second(time_str_to_time(start)), time_to_second(time_str_to_time(end))
        case "sample":
            assert sr is not None, "sr must be provided when ret_time_type is 'samples'"
            return prefix, time_to_samples(time_str_to_time(start), sr), time_to_samples(time_str_to_time(end), sr)


if __name__ == "__main__":
    seconds = 10000.25
    sr = 22500
    
    print(f"Seconds: {seconds}")
    
    print(f"Samples: {second_to_samples(seconds, sr)}")
    print(f"Samples: {time_to_samples(second_to_time(seconds), sr)}")
    
    print(f"Time: {second_to_time(seconds)}")
    print(f"Time: {samples_to_time(second_to_samples(seconds, sr), sr)}")
    print(f"Time: {time_str_to_time(time_to_name_str(second_to_time(seconds)))}")
    
    print(f"Name: {time_to_name_str(second_to_time(seconds))}")