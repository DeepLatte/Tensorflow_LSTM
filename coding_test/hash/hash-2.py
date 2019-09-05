import collections

def solution(phone_book):
    answer = True
    for number in phone_book:
        phone_book2 = list(phone_book)
        phone_book2.remove(number)
        len_numb = len(number)
        res = [number2[:len_numb] for number2 in phone_book2]
        c = collections.Counter(res)
        if c[number] > 0:
            return False

    return answer


a = ['123','456','789']

res = solution(a)
print(res)

# c = collections.Counter(a=2, b=3, c=2) # 값=개수 형태로 입력이 가능함
# print(collections.Counter(c))          # Counter({'b': 3, 'c': 2, 'a': 2})
# print(sorted(c.elements())             # ['a', 'a', 'b', 'b', 'b', 'c', 'c'], 요소(key)에 해당하는 값을 풀어서 반환

# container = collections.Counter()
# container.update("aabcdeffgg")         #  값을 갱신하는 것을 의미한다, 딕셔너리 형태로도 업데이트가 가능함.
# print(container)                         # Counter({'f': 2, 'g': 2, 'a': 2, 'e': 1, 'b': 1, 'c': 1, 'd': 1})
#
# for k,v in container.items():
#     print(k,':',v)

# Counter를 통한 subtract
# c3 = collections.Counter('hello python')
# c4 = collections.Counter('i love python')
# c3.subtract(c4)
# print(c3)                              # Counter({'l': 1, 'h': 1, 'n': 0, 't': 0, 'p': 0, 'e': 0, 'o': 0, 'y': 0, 'i': -1, 'v': -1, ' ': -1})
# 이 외에도 +, -. &, |(합집합) 이 가능하다. 뺄셈에서 음수값은 출력되지 않는다.

