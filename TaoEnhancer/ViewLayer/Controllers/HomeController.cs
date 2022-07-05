using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using ViewLayer.Models;

namespace ViewLayer.Controllers
{
    public class HomeController : Controller
    {
        // private ItemController itemController = new ItemController();
        // private StudentController studentController = new StudentController();
        private TestController testController = new TestController();

        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult TeacherMenu()
        {
            return View();
        }

        public IActionResult StudentMenu()
        {
            return View();
        }

        public IActionResult ManageSolvedTestList()
        {
            return View();
        }

        public IActionResult TestTemplateList()
        {
            return View(new TestTemplateListModel
            {
                TestTemplates = testController.LoadTestTemplates()
            });
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}