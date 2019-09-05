import collections

def solution(participant, completion):
    res = { completion[0] : 1 }
    for idx, name in enumerate(completion[1:]):
        if name in res:
            res[name] += 1
        else:
            res[name] = 1
    for idx, name in enumerate(participant):
        if name not in res:
            return name
        # if res[name] is 1:
        #     return name
        res[name] -= 1
    val = res.values()
    for i, v in enumerate(val):
        if v is not 0:
            k = list(res.keys())
            return res[k[i]]
    answer= 0
    return answer

def solution2(participant, completion):
    # collections 를 사용한 다른 솔루션

    p = collections.Counter(participant)
    c = collections.Counter(completion)

    p.subtract(c)

    answer = (p.most_common(1))[0][0]

    return answer

a = ["mislav", "stanko", "mislav", "ana"]
b = ["stanko", "ana", "mislav"]

res = solution(a,b)