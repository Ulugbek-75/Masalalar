# 1920
class Solution:
    def buildArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n

        for i in range(n):
            ans[i] = nums[nums[i]]

        return ans
# 1108
def defang_ip_address(address):
    defanged_address = address.replace('.', '[.]')
    return defanged_address
    

# 9
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        str_x = str(x)
        return str_x == str_x[::-1]


# 3
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_index_map = {}
        start = 0
        max_length = 0

        for end, char in enumerate(s):
            if char in char_index_map and char_index_map[char] >= start:
                start = char_index_map[char] + 1

            char_index_map[char] = end
            max_length = max(max_length, end - start + 1)

        return max_length



# 1
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_indices = {}

        for i, num in enumerate(nums):
            complement = target - num

            if complement in num_indices:
                return [num_indices[complement], i]

            num_indices[num] = i

        return []


# 136
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0

        for num in nums:
            result ^= num

        return result



# 137
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0

        for i in range(32):
            count = 0
            for num in nums:
                count += (num >> i) & 1

            result |= (count % 3) << i

        if result & (1 << 31):
            result -= (1 << 32)

        return result




# 27
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        k = 0  # Variable to store the length of the modified array

        for num in nums:
            if num != val:
                nums[k] = num
                k += 1

        return k


# 28
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        return haystack.find(needle)



# 29
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        INT_MAX = 2 ** 31 - 1
        INT_MIN = -2 ** 31

        if dividend == 0:
            return 0

        sign = -1 if (dividend < 0) ^ (divisor < 0) else 1

        dividend, divisor = abs(dividend), abs(divisor)

        quotient = 0

        while dividend >= divisor:
            dividend -= divisor
            quotient += 1

        result = sign * quotient

        if result > INT_MAX:
            return INT_MAX
        elif result < INT_MIN:
            return INT_MIN
        else:
            return result
