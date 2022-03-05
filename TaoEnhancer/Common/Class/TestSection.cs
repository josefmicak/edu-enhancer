namespace Common.Class
{
    public class TestSection
    {
        private string pIdentifier;
        private bool pRequired;
        private bool pFixed;
        private string pTitle;
        private bool pVisible;
        private bool pKeepTogether;
        private TestSessionControl? pSessionControl;
        private TestTimeLimits? pTimeLimits;
        private List<TestItem> pItems = new List<TestItem>();

        public string Identifier { get { return pIdentifier; } }
        public bool Required { get { return pRequired; } }
        public bool Fixed { get { return pFixed; } }
        public string Title { get { return pTitle; } }
        public bool Visible { get { return pVisible; } }
        public bool KeepTogether { get { return pKeepTogether; } }
        public TestSessionControl? SessionControl { set { pSessionControl = value; } get { return pSessionControl; } }
        public TestTimeLimits? TimeLimits { set { pTimeLimits = value; } get { return pTimeLimits; } }
        public List<TestItem> Items { set { pItems = value; } get { return pItems; } }

        public TestSection(string aIdentifier, bool aRequired, bool aFixed, string aTitle, bool aVisible, bool aKeepTogether)
        {
            pIdentifier = aIdentifier;
            pRequired = aRequired;
            pFixed = aFixed;
            pTitle = aTitle;
            pVisible = aVisible;
            pKeepTogether = aKeepTogether;
        }

        public override string ToString()
        {
            string items = "";
            foreach (TestItem item in Items)
            {
                items += "{" + item + "}, ";
            }

            return
                "Identifier: " + Identifier + ", " +
                "Required: " + Required + ", " +
                "Fixed: " + Fixed + ", " +
                "Title: " + Title + ", " +
                "Visible: " + Visible + ", " +
                "KeepTogether: " + KeepTogether + ", " +
                "SessionControl: {" + SessionControl + "}, " +
                "TimeLimits: {" + TimeLimits + "}, " +
                "Items: [" + items.Substring(0, items.Length - 2) + "]";
        }
    }
}
