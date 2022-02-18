namespace Common.Class
{
    public class TestTimeLimits
    {
        private bool pAllowLateSubmission;

        public bool AllowLateSubmission { get { return pAllowLateSubmission; } }

        public TestTimeLimits(bool aAllowLateSubmission)
        {
            pAllowLateSubmission = aAllowLateSubmission;
        }

        public override string ToString()
        {
            return
                "AllowLateSubmission: " + AllowLateSubmission;
        }
    }
}
