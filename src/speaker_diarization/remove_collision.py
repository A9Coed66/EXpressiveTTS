def remove_collision(diarization):
    logs = []
    pre = None
    last_end = None
    is_collision = False
    # TODO: kiểm tra lại xem có đúng tiền đề là đã sort hay chưa
    # TODO: đưa ra thuật toán tối ưu hơn
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Initialize previous record
        if not pre:
            pre = [round(turn.start,2), round(turn.end,2)]
            last_end = turn.end
        else:
            # Start collision
            if turn.start<last_end:
                is_collision = True
                last_end = max(last_end, turn.end)
            else:
                if is_collision:
                    is_collision = False
                    last_end = turn.end
                    pre = [round(turn.start,2), round(turn.end,2)]
                else:
                    last_end = turn.end
                    logs.append(pre)
                    pre = [round(turn.start,2), round(turn.end,2)]
    return logs
