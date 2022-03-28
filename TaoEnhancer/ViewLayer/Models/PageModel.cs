namespace ViewLayer.Models
{
    public class PageModel
    {
        private string title;
        private int userRole;

        public string Solution { get { return "TaoEnhancer"; } }
        public string Title { get { return title; } set { title = value; } }
        public int UserRole { get { return userRole; } set { userRole = value; } }
    }
}