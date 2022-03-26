namespace ViewLayer.Models
{
    public class ManageSolvedItemModel : BrowseSolvedItemModel
    {
        private int currentSubitemIndex;
        private string studentsPoints;
        private string errorText;

        public int CurrentSubitemIndex { get { return currentSubitemIndex; } set { currentSubitemIndex = value; } }
        public string StudentsPoints { get { return studentsPoints; } set { studentsPoints = value; } }
        public string ErrorText { get { return errorText; } set { errorText = value; } }
    }
}