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
    
