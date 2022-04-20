namespace ViewLayer.Models
{
    public class PageModel
    {
        private string title;
        private (string message, string messageClass) headerMessageData;
        private int userRole;

        public string Solution { get { return "TaoEnhancer"; } }
        public string Title { get { return title; } set { title = value; } }
        public (string message, string messageClass) HeaderMessageData { get { return headerMessageData; } set { headerMessageData = value; } }
        public int UserRole { get { return userRole; } set { userRole = value; } }
    }
}