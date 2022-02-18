namespace Common.Class
{
    public class TestPart
    {
        private string pIdentifier;
        private string pNavigationMode;
        private string pSubmissionMode;
        private TestSessionControl pSessionControl;
        private TestTimeLimits pTimeLimits;
        private List<TestSection> pSections = new List<TestSection>();

        public string Identifier { get { return pIdentifier; } }
        public string NavigationMode { get { return pNavigationMode; } }
        public string SubmissionMode { get { return pSubmissionMode; } }
        public TestSessionControl SessionControl { set { pSessionControl = value; } get { return pSessionControl; } }
        public TestTimeLimits TimeLimits { set { pTimeLimits = value; } get { return pTimeLimits; } }
        public List<TestSection> Sections { set { pSections = value; } get { return pSections; } }

        public TestPart(string aIdentifier, string aNavigationMode, string aSubmissionMode)
        {
            pIdentifier = aIdentifier;
            pNavigationMode = aNavigationMode;
            pSubmissionMode = aSubmissionMode;
        }

        public override string ToString()
        {
            string sections = "";
            foreach (TestSection section in Sections)
            {
                sections += "{" + section + "}, ";
            }

            return
                "Identifier: " + Identifier + ", " +
                "NavigationMode: " + NavigationMode + ", " +
                "SubmissionMode: " + SubmissionMode + ", " +
                "SessionControl: {" + SessionControl + "}, " +
                "TimeLimits: {" + TimeLimits + "}, " +
                "Sections: [" + sections.Substring(0, sections.Length - 2) + "]";
        }
    }
}
