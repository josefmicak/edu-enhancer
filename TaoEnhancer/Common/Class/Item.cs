namespace Common.Class
{
    public class Item
    {
        private string pIdentifier;
        private string pTitle;
        private string pLabel;
        private bool pAdaptive;
        private bool pTimeDependent;
        private string pToolName;
        private string pToolVersion;

        public string Identifier { get { return pIdentifier; } }
        public string Title { get { return pTitle; } }
        public string Label { get { return pLabel; } }
        public bool Adaptive { get { return pAdaptive; } }
        public bool TimeDependent { get { return pTimeDependent; } }
        public string ToolName { get { return pToolName; } }
        public string ToolVersion { get { return pToolVersion; } }

        public Item(string aIdentifier, string aTitle, string aLabel, bool aAdaptive, bool aTimeDependent, string aToolName, string aToolVersion)
        {
            pIdentifier = aIdentifier;
            pTitle = aTitle;
            pLabel = aLabel;
            pAdaptive = aAdaptive;
            pTimeDependent = aTimeDependent;
            pToolName = aToolName;
            pToolVersion = aToolVersion;
        }

        public override string ToString()
        {
            return
                "Item: {" +
                    "Identifier: " + Identifier + ", " +
                    "Title: " + Title + ", " +
                    "Label: " + Label + ", " +
                    "Adaptive: " + Adaptive + ", " +
                    "TimeDependent: " + TimeDependent + ", " +
                    "ToolName: " + ToolName + ", " +
                    "ToolVersion: " + ToolVersion +
                "}";
        }
    }
}
