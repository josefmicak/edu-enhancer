namespace ViewLayer.Models
{
    public class IndexModel : PageModel
    {
        private string text;
        private string textClass;
        private string signInURL;

        public string Text { get { return text; } set { text = value; } }
        public string TextClass { get { return textClass; } set { textClass = value; } }
        public string SignInURL { get { return signInURL; } set { signInURL = value; } }
    }
}