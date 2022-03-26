namespace ViewLayer.Models
{
    public class IndexModel : PageModel
    {
        private string signInURL;

        public string SignInURL { get { return signInURL; } set { signInURL = value; } }
    }
}