namespace Common.Class
{
    public class Result
    {
        private string pIdentifier;
        private string pDatestamp;

        public string Identifier { set { pIdentifier = value; } get { return pIdentifier; } }
        public string Datestamp { set { pDatestamp = value; } get { return pDatestamp; } }

        public Result() { }

        public override string ToString()
        {
            return "";
        }
    }
}
