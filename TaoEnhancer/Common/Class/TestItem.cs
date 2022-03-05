namespace Common.Class
{
    public class TestItem
    {
        private string pIdentifier;
        private bool pRequired;
        private bool pFixed;
        private string pHref;
        private TestSessionControl? pSessionControl;
        private TestTimeLimits? pTimeLimits;
        private bool pPointsDetermined = false;
        private int pPoints = 0;

        public string Identifier { get { return pIdentifier; } }
        public bool Required { get { return pRequired; } }
        public bool Fixed { get { return pFixed; } }
        public string Href { get { return pHref; } }
        public TestSessionControl? SessionControl { set { pSessionControl = value; } get { return pSessionControl; } }
        public TestTimeLimits? TimeLimits { set { pTimeLimits = value; } get { return pTimeLimits; } }
        public bool PointsDetermined { set { pPointsDetermined = value; } get { return pPointsDetermined; } }
        public int Points { set { pPoints = value; } get { return pPoints; } }
        public string NumberIdentifier { get { return pHref.Split("/")[3]; } }

        public TestItem(string aIdentifier, bool aRequired, bool aFixed, string aHref)
        {
            pIdentifier = aIdentifier;
            pRequired = aRequired;
            pFixed = aFixed;
            pHref = aHref;
        }

        public override string ToString()
        {
            return
                "Identifier: " + Identifier + ", " +
                "Required: " + Required + ", " +
                "Fixed: " + Fixed + ", " +
                "Href: " + Href + ", " +
                "SessionControl: {" + SessionControl + "}, " +
                "TimeLimits: {" + TimeLimits + "}, " +
                "PointsDetermined: " + PointsDetermined + ", " +
                "Points: " + Points;
        }
    }
}
