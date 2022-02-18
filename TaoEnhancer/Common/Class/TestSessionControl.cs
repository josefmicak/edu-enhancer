namespace Common.Class
{
    public class TestSessionControl
    {
        private int pMaxAttempts;
        private bool pShowFeedback;
        private bool pAllowReview;
        private bool pShowSolution;
        private bool pAllowComment;
        private bool pAllowSkipping;
        private bool pValidateResponses;

        public int MaxAttempts { get { return pMaxAttempts; } }
        public bool ShowFeedback { get { return pShowFeedback; } }
        public bool AllowReview { get { return pAllowReview; } }
        public bool ShowSolution { get { return pShowSolution; } }
        public bool AllowComment { get { return pAllowComment; } }
        public bool AllowSkipping { get { return pAllowSkipping; } }
        public bool ValidateResponses { get { return pValidateResponses; } }

        public TestSessionControl(int aMaxAttempts, bool aShowFeedback, bool aAllowReview, bool aShowSolution, bool aAllowComment, bool aAllowSkipping, bool aValidateResponses)
        {
            pMaxAttempts = aMaxAttempts;
            pShowFeedback = aShowFeedback;
            pAllowReview = aAllowReview;
            pShowSolution = aShowSolution;
            pAllowComment = aAllowComment;
            pAllowSkipping = aAllowSkipping;
            pValidateResponses = aValidateResponses;
        }

        public override string ToString()
        {
            return
                "MaxAttempts: " + MaxAttempts + ", " +
                "ShowFeedback: " + ShowFeedback + ", " +
                "AllowReview: " + AllowReview + ", " +
                "ShowSolution: " + ShowSolution + ", " +
                "AllowComment: " + AllowComment + ", " +
                "AllowSkipping: " + AllowSkipping + ", " +
                "ValidateResponses: " + ValidateResponses;
        }
    }
}
