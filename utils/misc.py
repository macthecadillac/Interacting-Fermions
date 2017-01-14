"""This module provides tools that are useful but not, strictly
speaking, related to quantum mechanics. Function included:
    bin_to_dec
    permutation
    binary_permutation
    permutation_any_order

1-13-2017
"""

def bin_to_dec(l):
    """Converts a list "l" of 1s and 0s into a decimal"""
    return int(''.join(map(str, l)), 2)


def permutation(l):
    """
    Code plagiarized from StackOverflow. With a given list of values,
    this function changes the list in situ to the next permutation.
    This function differs from itertools.permutations in that it takes
    into account repeated values in the list and avoids returning duplicates.

    Args: "l" is the list of elements we wish to permute
          "o" is the permutation order
    Returns: a list
    """
    n = len(l)
    # Step 1: Find tail
    last = n - 1  # tail is from `last` to end
    while last > 0:
        if l[last - 1] < l[last]:
            break
        last -= 1
    # Step 2: Increase the number just before tail
    if last > 0:
        small = l[last - 1]
        big = n - 1
        while l[big] <= small:
            big -= 1
        l[last - 1], l[big] = l[big], small
    # Step 3: Reverse tail
    i = last
    j = n - 1
    while i < j:
        l[i], l[j] = l[j], l[i]
        i += 1
        j -= 1
    return l


def binary_permutation(l):
    """
    Find the next permutation of a list of ones and zeros. This function
    permutes in the reverse order of next_permutation.

    Args: "l" is the list of elements we wish to permute
          "o" is the permutation order
    Returns: a list
    """
    n = len(l) - 1
    migrate = False
    while True:
        # Find the last 1
        i = n
        while True:
            if l[i] == 1:
                break
            else:
                i -= 1
        # Switch the element with the next element if the element is
        #  not the last element.
        if i != n:
            l[i], l[i + 1] = l[i + 1], l[i]
            if migrate:
                i += 2
                j = i
                # Find the first 1 to the right of the 1 we just moved.
                while True:
                    if l[j] == 1:
                        break
                    else:
                        j += 1
                        if j >= len(l):
                            break
                # Move all the 1's to the left.
                w = len(l[j:])
                for k in range(w):
                    l[i], l[j] = l[j], l[i]
                    i += 1
                    j += 1
                migrate = False
            break
        # Since there is a 1/some 1's at the very end of the list,
        #  we loop to look for the next one to the left that is
        #  separated by some 0's.
        else:
            # A flag to tell the function to move all the 1's at
            #  the right end to the left.
            migrate = True
            n -= 1
    return l


def permutation_any_order(l, o='lexicographic'):
    """The slowest permutation solution of all but the most versatile.
    (About 10x slower than the other two) This function is capable
    of permuting in either the lexicographical order or the
    anti-lexicographical order. It can also permute lists of all kinds
    of objects, not only numbers.

    Args: "l" is the list of elements we wish to permute
          "o" is the permutation order
    Returns: a list
    """
    # Flags to change the lexicographical order when permuting
    f1 = True if o == 'lexicographic' else False
    f2 = True if o == 'antilexicographic' else False

    for s1 in range(1, len(l)):  # Index of term from the end of list -1
        if l[-1 - s1:] == sorted(l[-1 - s1:], reverse=f1):
            continue
        else:
            for s2 in range(1, s1 + 1):  # Index of term from the end of list -1
                if l[-1 - s2:] == sorted(l[-1 - s2:], reverse=f1):
                    continue
                else:
                    l[-s2:] = sorted(l[-s2:], reverse=f2)
                    for i, n in enumerate(l[-s2:]):
                        if (f1 and n > l[-1 - s2]) or (f2 and n < l[-1 - s2]):
                            l[-1 - s2], l[i - s2] = l[i - s2], l[-1 - s2]
                            return l
    else:
        l.sort(reverse=f2)
        return l
