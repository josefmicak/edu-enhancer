namespace Common.Class
{
    public class ItemResponse
    {
        private string pIdentifier;
        private string pCorrectChoicePoints = "0";
        private string pWrongChoicePoints = "0";
        
        public string Identifier { get { return pIdentifier; } }
        public int CorrectChoicePoints {
            get 
            {
                if (pCorrectChoicePoints != "N/A")
                {
                    return int.Parse(pCorrectChoicePoints);
                }
                return 0;
            }
        }
        public double WrongChoicePoints {
            get
            {
                if (pWrongChoicePoints != "N/A")
                {
                    return double.Parse(pWrongChoicePoints);
                }
                return 0;
            }
        }
        public bool RecommendedWrongChoicePoints {
            get
            {
                return (WrongChoicePoints == CorrectChoicePoints * -1);
            }
        }
        public bool PointsDetermined
        {
            get
            {
                return (pCorrectChoicePoints != "N/A");
            }
        }

        public ItemResponse(string aIdentifier, string aCorrectChoicePoints, string aWrongChoicePoints)
        {
            pIdentifier = aIdentifier;
            pCorrectChoicePoints = aCorrectChoicePoints;
            pWrongChoicePoints = aWrongChoicePoints;
        }

        public ItemResponse(string aIdentifier, string aCorrectChoicePoints)
        {
            pIdentifier = aIdentifier;
            pCorrectChoicePoints = aCorrectChoicePoints;
        }
    }
}
