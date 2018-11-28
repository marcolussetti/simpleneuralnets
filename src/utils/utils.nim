import math, sugar

proc unzip*[S, T](input: openArray[tuple[a: S, b: T]]): tuple[a: seq[S],
    b: seq[T]] =
    var s1: seq[S] = newSeq[S]()
    var s2: seq[T] = newSeq[T]()

    for t in input:
        s1.add(t[0]); s2.add(t[1])

    return (s1, s2)

proc `/`*(s: seq[int], d: float): seq[float] {.inline.} =
    return lc[i.float / d | (i <- s), float]

proc maxIndex*(arr: seq[float]): int =
    assert arr.len > 0
    result = 0
    for i in 1..<arr.len:
        if arr[i] > arr[result]:
            result = i

