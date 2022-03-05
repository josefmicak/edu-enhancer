namespace Common.Class
{
    public class Item
    {
        private string pIdentifier;
        private string pTitle;
        private string pLabel;
        private bool pAdaptive;
        private bool pTimeDependent;
        //private string pToolName;
        //private string pToolVersion;
        private bool pPointsFileExists;
        private string pNumberIdentifier;
        private List<ItemResponse> pResponses = new List<ItemResponse>();
        
        public string Identifier { get { return pIdentifier; } }
        public string Title { get { return pTitle; } }
        public string Label { get { return pLabel; } }
        public bool Adaptive { get { return pAdaptive; } }
        public bool TimeDependent { get { return pTimeDependent; } }
        //public string ToolName { get { return pToolName; } }
        //public string ToolVersion { get { return pToolVersion; } }
        public bool PointsFileExists { get { return pPointsFileExists; } }
        public string NumberIdentifier { set { pNumberIdentifier = value; } get { return pNumberIdentifier; } }
        public List<ItemResponse> Responses { set { pResponses = value; } get { return pResponses; } }
        public int Points {
            get
            {
                if (pPointsFileExists)
                {
                    int points = 0;
                    foreach (ItemResponse response in Responses)
                    {
                        if (response.PointsDetermined)
                        {
                            points += response.CorrectChoicePoints;
                        }
                    }
                    return points;
                }
                return 0;
            }
        }
        public bool PointsDetermined
        {
            get
            {
                if (pPointsFileExists)
                {
                    foreach (ItemResponse response in Responses)
                    {
                        if (!response.PointsDetermined)
                        {
                            return false;
                        }
                    }
                    return true;
                }
                return false;
            }
        }

        public Item(string aIdentifier, string aTitle, string aLabel, bool aAdaptive, bool aTimeDependent/*, string aToolName, string aToolVersion*/, bool aPointsFileExists, string aNumberIdentifier)
        {
            pIdentifier = aIdentifier;
            pTitle = aTitle;
            pLabel = aLabel;
            pAdaptive = aAdaptive;
            pTimeDependent = aTimeDependent;
            //pToolName = aToolName;
            //pToolVersion = aToolVersion;
            pPointsFileExists = aPointsFileExists;
            pNumberIdentifier = aNumberIdentifier;
        }

        public override string ToString()
        {
            return
                "Item: {" +
                    "Identifier: " + Identifier + ", " +
                    "Title: " + Title + ", " +
                    "Label: " + Label + ", " +
                    "Adaptive: " + Adaptive + ", " +
                    "TimeDependent: " + TimeDependent +// ", " +
                    //"ToolName: " + ToolName + ", " +
                    //"ToolVersion: " + ToolVersion +
                "}";
        }
    }
}
