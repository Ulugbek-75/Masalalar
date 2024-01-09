


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




# 41
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)

        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i]-1] != nums[i]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        return n + 1

# 42
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height or len(height) < 3:
            return 0

        n = len(height)
        left_max, right_max = [0] * n, [0] * n
        left_max[0] = height[0]
        right_max[n - 1] = height[n - 1]

        for i in range(1, n):
            left_max[i] = max(left_max[i - 1], height[i])

        for i in range(n - 2, -1, -1):
            right_max[i] = max(right_max[i + 1], height[i])

        trapped_water = 0

        for i in range(n):
            trapped_water += max(0, min(left_max[i], right_max[i]) - height[i])

        return trapped_water



# 141
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False

        sekin = head
        tez = head.next

        while sekin != tez:
            if not tez or not tez.next:
                return False
            slow = slow.next
            fast = fast.next.next

        return True


# 142
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return None

        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None 

        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next

        return slow


# 205
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        s_to_t = {}
        t_to_s = {}

        for char_s, char_t in zip(s, t):
            if char_s in s_to_t and s_to_t[char_s] != char_t:
                return False
            if char_t in t_to_s and t_to_s[char_t] != char_s:
                return False

            s_to_t[char_s] = char_t
            t_to_s[char_t] = char_s

        return True

# 206
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head

        while current is not None:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

        return prev



# 209
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        start = 0
        end = 0
        min_length = float('inf')
        current_sum = 0

        while end < n:
            current_sum += nums[end]

            while current_sum >= target:
                min_length = min(min_length, end - start + 1)
                current_sum -= nums[start]
                start += 1

            end += 1

        return min_length if min_length != float('inf') else 0


# 217
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        S = set()
        for num in nums:
            if num in S:
                return True
            S.add(num)
        return False

# 216
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def backtrack(start, target, path):
            if target == 0 and len(path) == k:
                result.append(path[:])
                return
            for i in range(start, 10):
                if i > target:
                    break
                path.append(i)
                backtrack(i + 1, target - i, path)
                path.pop()

        result = []
        backtrack(1, n, [])
        return result




# 455
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()

        content_children = 0
        child_idx, cookie_idx = 0, 0

        while child_idx < len(g) and cookie_idx < len(s):
            if s[cookie_idx] >= g[child_idx]:

                content_children += 1
                child_idx += 1 
            cookie_idx += 1  

        return content_children



# 1913
class Solution:
    def maxProductDifference(self, nums: List[int]) -> int:
        nums.sort()

        max_difference = (nums[-1] * nums[-2]) - (nums[0] * nums[1])

        return max_difference



# 1903
class Solution:
    def largestOddNumber(self, num: str) -> str:
        for i in range(len(num) - 1, -1, -1):
            if int(num[i]) % 2 == 1:
                return num[:i + 1]
        return ""
# 872
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def get_leaf_values(node, leaves):
            if not node:
                return

            if not node.left and not node.right:
                leaves.append(node.val)

            get_leaf_values(node.left, leaves)
            get_leaf_values(node.right, leaves)

        leaves1, leaves2 = [], []

        get_leaf_values(root1, leaves1)
        get_leaf_values(root2, leaves2)

        return leaves1 == leaves2 
