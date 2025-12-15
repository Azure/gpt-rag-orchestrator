from semantic_kernel.functions import kernel_function
from datetime import datetime

class CommonPlugin:

    @kernel_function(name="GetTodayDate",
                 description="The output is today's date in string format")
    def get_today_date(self) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        return f"Today's date is {today}"

    @kernel_function(name="GetTime",
                 description="The output is the current time (hour and minutes) in HH:MM format")
    def get_time(self) -> str:
        current_time = datetime.now().strftime("%H:%M")
        return f"The current time is {current_time}"