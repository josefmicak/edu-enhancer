namespace Common.Class
{
    public class Test
    {
        private string pIdentifier;
        private string pTitle;
        private string pToolName;
        private string pToolVersion;
        private TestSessionControl pSessionControl;
        private TestTimeLimits pTimeLimits;
        private List<TestPart> pParts = new List<TestPart>();
        private bool pPointsDetermined = true;
        private bool pNegativePoints = false;

        public string Identifier { get { return pIdentifier; } }
        public string Title { get { return pTitle; } }
        public string ToolName { get { return pToolName; } }
        public string ToolVersion { get { return pToolVersion; } }
        public TestSessionControl SessionControl { set { pSessionControl = value; } get { return pSessionControl; } }
        public TestTimeLimits TimeLimits { set { pTimeLimits = value; } get { return pTimeLimits; } }
        public List<TestPart> Parts { set { pParts = value; } get { return pParts; } }
        public bool PointsDetermined { set { pPointsDetermined = value; } get { return pPointsDetermined; } }
        public bool NegativePoints { set { pNegativePoints = value; } get { return pNegativePoints; } }
        public List<TestItem> Items
        {
            get
            {
                List<TestItem> items = new List<TestItem>();
                foreach (TestPart part in Parts)
                {
                    foreach (TestSection section in part.Sections)
                    {
                        foreach (TestItem item in section.Items)
                        {
                            items.Add(item);
                        }
                    }
                }

                return items;
            }
        }
        public int Points
        {
            get
            {
                int points = 0;
                foreach (TestPart part in Parts)
                {
                    foreach (TestSection section in part.Sections)
                    {
                        foreach (TestItem item in section.Items)
                        {
                            if (item.PointsDetermined)
                            {
                                points += item.Points;
                            }
                        }
                    }
                }

                return points;
            }
        }

        public Test(string aIdentifier, string aTitle, string aToolName, string aToolVersion)
        {
            pIdentifier = aIdentifier;
            pTitle = aTitle;
            pToolName = aToolName;
            pToolVersion = aToolVersion;
        }

        public override string ToString()
        {
            string parts = "";
            foreach (TestPart part in Parts)
            {
                parts += "{" + part + "}, ";
            }

            return
                "Test: {" +
                    "Identifier: " + Identifier + ", " +
                    "Title: " + Title + ", " +
                    "ToolName: " + ToolName + ", " +
                    "ToolVersion: " + ToolVersion + ", " +
                    "SessionControl: {" + SessionControl + "}, " +
                    "TimeLimits: {" + TimeLimits + "}, " +
                    "Parts: [" + parts.Substring(0, parts.Length - 2) + "], " +
                    "PointsDetermined: " + PointsDetermined + ", " +
                    "NegativePoints: " + NegativePoints + ", " +
                    "Items: " + Items.Count + ", " +
                    "Points: " + Points +
                "}";
        }
    }
}
